import threading
from collections import deque
import numpy as np
from aec_audio_processing import AudioProcessor

class WebRTCAcousticEchoCanceller:
    """Thread-safe Acoustic Echo Cancellation wrapper around WebRTC AudioProcessor.
    
    Combines linear WebRTC AEC with a non-linear energy ducking safety gate.
    Maintains a rolling buffer of system loopback (reference) audio and aligns it
    with the microphone audio based on a configurable render-to-capture delay.
    Processes audio in WebRTC's native 10ms frame sizes.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        delay_ms: int = 80,
        enable_ns: bool = False,
        enable_agc: bool = False,
        ducking_threshold: float = 0.45,
        ref_threshold: float = 0.01,
        correlation_threshold: float = 0.30,
        correlation_lag_search_ms: int = 60,
        correlation_window_ms: int = 32,
    ):
        self.sample_rate = sample_rate
        self.delay_sec = delay_ms / 1000.0
        self.ducking_threshold = ducking_threshold
        self.ref_threshold = ref_threshold
        # Correlation-based ducking: catches loud residual echo (e.g. speakers close to
        # an open mic, no headphones) that the ratio+absolute-cap ducking below misses,
        # because that check refuses to duck anything with mic_rms >= 0.05 - which is
        # exactly the regime a strong-but-real leak sits in. Correlation directly answers
        # "is this mic waveform explained by what's currently playing on the speaker",
        # which is robust to loudness in a way an RMS ratio isn't.
        #
        # KNOWN LIMITATION: this is tuned to reliably kill *pure* leak (mic picks up only
        # the speaker output, no one talking - the confirmed, reported failure mode: a
        # YouTube clip played with no one speaking produced hallucinated "Speaker 1" text).
        # It is not a solved general double-talk detector - genuine speech that happens to
        # occur *while* system audio is also playing can still get partially suppressed,
        # since distinguishing "residual echo" from "real speech overlapping a leak" from
        # short-window correlation alone is a genuinely hard, actively-researched AEC
        # problem (see e.g. how much engineering Zoom/Teams put into this). Wearing
        # headphones remains the reliable fix for that case - see README.
        self.correlation_threshold = correlation_threshold
        self.correlation_lag_search_samples = int(sample_rate * correlation_lag_search_ms / 1000)
        self.correlation_window_samples = int(sample_rate * correlation_window_ms / 1000)
        self._residual_buffer: deque = deque()
        self._residual_buffer_start_idx: int | None = None
        
        # WebRTC APM instance
        self.ap = AudioProcessor(
            enable_aec=True,
            enable_ns=enable_ns,
            enable_agc=enable_agc,
            enable_vad=False
        )
        self.ap.set_stream_format(self.sample_rate, 1)
        self.ap.set_reverse_stream_format(self.sample_rate, 1)
        
        # WebRTC internal frame delay. We handle coarse alignment ourselves via timestamps,
        # so we set a low nominal internal delay (e.g., 40ms) for WebRTC's fine-tuning estimator.
        self.ap.set_stream_delay(40)
        
        self.frame_samples = int(self.sample_rate * 10 / 1000)  # 160 samples for 10ms @ 16kHz
        
        # Sliding buffer for reference audio
        self.ref_start_time = None
        self.all_ref_samples = np.empty(0, dtype=np.float32)
        self.ref_rms_history = []
        
        self.lock = threading.Lock()        
        # Prune when it exceeds 8 seconds, leaving last 3 seconds
        self.prune_threshold_samples = 8 * self.sample_rate
        self.prune_to_samples = 3 * self.sample_rate

        # Carryover buffers for WebRTC AEC frame alignment (10ms boundaries)
        self.mic_carryover = np.empty(0, dtype=np.float32)
        self.mic_carryover_time = None
        
        # Cleaned output carryover buffer to match VADSegmenter's frame size
        self.cleaned_carryover = np.empty(0, dtype=np.float32)
        self.cleaned_timestamp = None

    def push_reference(self, ref_chunk: np.ndarray, timestamp: float) -> None:
        """Pushes a chunk of speaker/loopback (reference) audio with its capture timestamp."""
        with self.lock:
            # Track loopback RMS energy history to recognize when Speaker 1 is talking
            ref_rms = np.sqrt(np.mean(ref_chunk**2)) if len(ref_chunk) > 0 else 0.0
            self.ref_rms_history.append(ref_rms)
            if len(self.ref_rms_history) > 15: # ~450ms sliding window
                self.ref_rms_history.pop(0)

            if self.ref_start_time is None:
                # First chunk dictates the start time of the reference stream
                self.ref_start_time = timestamp
                self.all_ref_samples = ref_chunk.copy()
            else:
                self.all_ref_samples = np.concatenate((self.all_ref_samples, ref_chunk))
            
            # Prune buffer to avoid memory leak
            if len(self.all_ref_samples) > self.prune_threshold_samples:
                num_to_prune = len(self.all_ref_samples) - self.prune_to_samples
                self.all_ref_samples = self.all_ref_samples[num_to_prune:]
                self.ref_start_time += num_to_prune / self.sample_rate

    def _extract_ref_window(self, start_idx: int, length: int) -> np.ndarray:
        """Returns `length` samples of reference audio starting at `start_idx`
        (may be negative / extend past the buffer), zero-padded where the
        buffer doesn't cover the requested range."""
        if start_idx < 0:
            missing_prefix = -start_idx
            if missing_prefix >= length:
                return np.zeros(length, dtype=np.float32)
            available = self.all_ref_samples[0: length - missing_prefix]
            return np.concatenate((np.zeros(missing_prefix, dtype=np.float32), available))
        end_idx = start_idx + length
        if end_idx <= len(self.all_ref_samples):
            return self.all_ref_samples[start_idx:end_idx].copy()
        available_samples = len(self.all_ref_samples) - start_idx
        if available_samples <= 0:
            return np.zeros(length, dtype=np.float32)
        available = self.all_ref_samples[start_idx:]
        missing_suffix = length - len(available)
        return np.concatenate((available, np.zeros(missing_suffix, dtype=np.float32)))

    def _max_correlation(self, mic_window: np.ndarray, start_idx: int) -> float:
        """Normalized cross-correlation of mic_window against the reference, searched
        over a small (narrow - see __init__ note) window of lags around start_idx, to
        correct for minor residual misalignment without over-searching."""
        win_len = len(mic_window)
        lag = self.correlation_lag_search_samples
        window = self._extract_ref_window(start_idx - lag, win_len + 2 * lag)

        mic_energy = float(np.dot(mic_window, mic_window))
        if mic_energy < 1e-9:
            return 0.0

        best = 0.0
        for offset in range(0, 2 * lag + 1):
            ref_slice = window[offset: offset + win_len]
            ref_energy = float(np.dot(ref_slice, ref_slice))
            if ref_energy < 1e-9:
                continue
            corr = abs(float(np.dot(mic_window, ref_slice))) / (mic_energy * ref_energy) ** 0.5
            if corr > best:
                best = corr
        return best

    def process_mic(self, mic_chunk: np.ndarray, mic_timestamp: float) -> tuple[np.ndarray, float]:
        """Aligns the mic chunk with reference history, applies carryover buffering for 10ms alignment,
        applies WebRTC AEC and ducking, and ensures the returned chunk matches the input chunk's length.
        """
        N = len(mic_chunk)
        if N == 0:
            return np.empty(0, dtype=np.float32), mic_timestamp

        with self.lock:
            # Initialize timestamps on first call
            if self.mic_carryover_time is None:
                self.mic_carryover_time = mic_timestamp
            if self.cleaned_timestamp is None:
                self.cleaned_timestamp = mic_timestamp

            # 1. Append mic_chunk to our input mic carryover buffer
            self.mic_carryover = np.concatenate((self.mic_carryover, mic_chunk))
            
            # Determine how many full 10ms frames we have
            total_samples = len(self.mic_carryover)
            num_frames = total_samples // self.frame_samples
            process_len = num_frames * self.frame_samples
            
            if process_len == 0:
                # Not enough samples for a 10ms frame; check if we can return any already-buffered cleaned samples
                if len(self.cleaned_carryover) >= N:
                    chunk_to_return = self.cleaned_carryover[:N]
                    self.cleaned_carryover = self.cleaned_carryover[N:]
                    ret_timestamp = self.cleaned_timestamp
                    self.cleaned_timestamp += N / self.sample_rate
                    return chunk_to_return, ret_timestamp
                return np.empty(0, dtype=np.float32), mic_timestamp
                
            # Extract the portion we will process
            mic_to_process = self.mic_carryover[:process_len]
            
            # Keep the remainder for the next call
            self.mic_carryover = self.mic_carryover[process_len:]
            
            # The timestamp for the start of this processed window
            processed_timestamp = self.mic_carryover_time
            
            # Update carryover start time for the next call
            self.mic_carryover_time += process_len / self.sample_rate
            
            # Find corresponding reference audio from history
            if self.ref_start_time is None or len(self.all_ref_samples) == 0:
                # No reference audio captured yet; treat as clean pass-through
                self.cleaned_carryover = np.concatenate((self.cleaned_carryover, mic_to_process))
                if len(self.cleaned_carryover) >= N:
                    chunk_to_return = self.cleaned_carryover[:N]
                    self.cleaned_carryover = self.cleaned_carryover[N:]
                    ret_timestamp = self.cleaned_timestamp
                    self.cleaned_timestamp += N / self.sample_rate
                    return chunk_to_return, ret_timestamp
                return np.empty(0, dtype=np.float32), mic_timestamp

            target_time = processed_timestamp - self.delay_sec
            start_idx = int((target_time - self.ref_start_time) * self.sample_rate)
            ref_chunk = self._extract_ref_window(start_idx, process_len)

            # --- Cross-Channel Energy Ducking / Echo Suppression Gate ---
            mic_rms = np.sqrt(np.mean(mic_to_process**2)) if len(mic_to_process) > 0 else 0.0
            max_ref_rms = max(self.ref_rms_history) if self.ref_rms_history else 0.0

            # If system speaker is active and mic energy indicates pure echo leakage
            if max_ref_rms > self.ref_threshold and (mic_rms / max_ref_rms) < self.ducking_threshold and mic_rms < 0.05:
                # Suppress the echo completely (setting to zero tells VAD it is silent)
                self.cleaned_carryover = np.concatenate((self.cleaned_carryover, np.zeros_like(mic_to_process)))
                if len(self.cleaned_carryover) >= N:
                    chunk_to_return = self.cleaned_carryover[:N]
                    self.cleaned_carryover = self.cleaned_carryover[N:]
                    ret_timestamp = self.cleaned_timestamp
                    self.cleaned_timestamp += N / self.sample_rate
                    return chunk_to_return, ret_timestamp
                return np.empty(0, dtype=np.float32), mic_timestamp

        # Run WebRTC AEC in 10ms frames (WITHOUT holding lock)
        clean_samples = []

        # Convert float32 arrays (-1.0 to 1.0) to PCM16 signed integers
        mic_pcm = (np.clip(mic_to_process, -1.0, 1.0) * 32767).astype(np.int16)
        ref_pcm = (np.clip(ref_chunk, -1.0, 1.0) * 32767).astype(np.int16)

        for i in range(0, process_len, self.frame_samples):
            mic_frame = mic_pcm[i : i + self.frame_samples]
            ref_frame = ref_pcm[i : i + self.frame_samples]

            # Feed reference (loopback) to reverse stream
            self.ap.process_reverse_stream(ref_frame.tobytes())

            # Cancel echo in microphone stream
            clean_frame_bytes = self.ap.process_stream(mic_frame.tobytes())

            # Convert back to float32
            clean_frame = np.frombuffer(clean_frame_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            clean_samples.append(clean_frame)

        cleaned_chunk = np.concatenate(clean_samples)

        # Re-acquire lock for correlation veto, output buffering, and returning chunk
        with self.lock:
            max_ref_rms = max(self.ref_rms_history) if self.ref_rms_history else 0.0
            if max_ref_rms <= self.ref_threshold:
                self._residual_buffer.clear()
                self._residual_buffer_start_idx = None
                self.cleaned_carryover = np.concatenate((self.cleaned_carryover, cleaned_chunk))
            else:
                if not self._residual_buffer:
                    self._residual_buffer_start_idx = start_idx
                self._residual_buffer.append(mic_to_process)

                buffered_len = sum(len(c) for c in self._residual_buffer)
                while buffered_len > self.correlation_window_samples and len(self._residual_buffer) > 1:
                    dropped = self._residual_buffer.popleft()
                    self._residual_buffer_start_idx += len(dropped)
                    buffered_len -= len(dropped)

                window_audio = np.concatenate(self._residual_buffer)
                window_rms = np.sqrt(np.mean(window_audio**2)) if len(window_audio) > 0 else 0.0
                if window_rms > 1e-6 and self._max_correlation(window_audio, self._residual_buffer_start_idx) > self.correlation_threshold:
                    self.cleaned_carryover = np.concatenate((self.cleaned_carryover, np.zeros_like(cleaned_chunk)))
                else:
                    self.cleaned_carryover = np.concatenate((self.cleaned_carryover, cleaned_chunk))

            # Return exactly N samples if we have enough
            if len(self.cleaned_carryover) >= N:
                chunk_to_return = self.cleaned_carryover[:N]
                self.cleaned_carryover = self.cleaned_carryover[N:]
                ret_timestamp = self.cleaned_timestamp
                self.cleaned_timestamp += N / self.sample_rate
                return chunk_to_return, ret_timestamp

            return np.empty(0, dtype=np.float32), mic_timestamp

    def measure_segment_correlation_and_rms(self, segment_audio: np.ndarray, absolute_start_time: float) -> tuple[float, float]:
        """Calculates the maximum normalized cross-correlation of the segment's audio
        against the reference history buffer at the corresponding timestamp range,
        along with the reference (speaker) audio's RMS value.
        Returns (max_correlation, ref_rms).
        """
        with self.lock:
            if self.ref_start_time is None or len(self.all_ref_samples) == 0:
                return 0.0, 0.0

            # Find the index in all_ref_samples corresponding to absolute_start_time
            start_idx = int((absolute_start_time - self.ref_start_time) * self.sample_rate)
            seg_len = len(segment_audio)

            # Extract the reference window with some search room (e.g. +/- 100ms lag)
            lag_search = int(self.sample_rate * 0.1) # 100ms
            ref_window = self._extract_ref_window(start_idx - lag_search, seg_len + 2 * lag_search)

            # Compute standard reference slice RMS (centered, without the lag padding)
            center_ref_slice = self._extract_ref_window(start_idx, seg_len)
            ref_rms = float(np.sqrt(np.mean(center_ref_slice**2))) if len(center_ref_slice) > 0 else 0.0

            seg_energy = float(np.dot(segment_audio, segment_audio))
            if seg_energy < 1e-9:
                return 0.0, ref_rms

            best_corr = 0.0
            # Search lags to align phase
            for offset in range(0, 2 * lag_search + 1):
                ref_slice = ref_window[offset : offset + seg_len]
                ref_energy = float(np.dot(ref_slice, ref_slice))
                if ref_energy < 1e-9:
                    continue
                corr = abs(float(np.dot(segment_audio, ref_slice))) / (seg_energy * ref_energy) ** 0.5
                if corr > best_corr:
                    best_corr = corr
            return best_corr, ref_rms



def calibrate_aec_delay(
    sample_rate: int = 16000,
    mic_device_name: str = None,
    speaker_device_name: str = None,
    calibration_duration_sec: float = 1.0,
) -> int:
    """Plays a chirp on the speaker while recording from both microphone and
    loopback speaker. Calculates the acoustic delay in ms by cross-correlating
    the recorded microphone and loopback audio. Returns the delay in ms,
    or falls back to 80ms if correlation peak is weak or fails.
    """
    import soundcard as sc
    import time
    import threading

    print(f"[AEC-Calibration] Starting calibration using mic: '{mic_device_name}', speaker: '{speaker_device_name}'...")

    # 1. Generate a chirp signal: 100ms linear sweep from 1000Hz to 4000Hz
    chirp_duration = 0.1
    t = np.linspace(0, chirp_duration, int(sample_rate * chirp_duration), endpoint=False)
    chirp = np.sin(2 * np.pi * (1000 + 3000 * t / chirp_duration) * t).astype(np.float32) * 0.4

    # Pad chirp with silence before and after
    silence_before = np.zeros(int(sample_rate * 0.1), dtype=np.float32)
    silence_after = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    play_signal = np.concatenate([silence_before, chirp, silence_after])

    # Get speaker device
    try:
        speaker = sc.get_speaker(speaker_device_name) if speaker_device_name else sc.default_speaker()
    except Exception as e:
        print(f"[AEC-Calibration] Warning: Speaker not available: {e}. Using default 80ms delay.")
        return 80

    # Get mic device
    try:
        mic = sc.get_microphone(mic_device_name) if mic_device_name else sc.default_microphone()
    except Exception as e:
        print(f"[AEC-Calibration] Warning: Microphone not available: {e}. Using default 80ms delay.")
        return 80

    # Get loopback device (same logic as SystemAudioCapture)
    loopback_mic = None
    try:
        loopback_mic = sc.get_microphone(speaker.id, include_loopback=True)
    except Exception:
        try:
            loopback_mic = sc.get_microphone(speaker.name, include_loopback=True)
        except Exception:
            try:
                default_spk = sc.default_speaker()
                loopback_mic = sc.get_microphone(default_spk.id, include_loopback=True)
            except Exception:
                pass

    if not loopback_mic:
        all_mics = sc.all_microphones(include_loopback=True)
        for m in all_mics:
            if m.id.startswith("{0.0.0.00000000}") or "loopback" in m.name.lower() or "monitor" in m.name.lower():
                loopback_mic = m
                break
        if not loopback_mic and all_mics:
            loopback_mic = all_mics[0]

    if not loopback_mic:
        print("[AEC-Calibration] Warning: Loopback device not found. Using default 80ms delay.")
        return 80

    mic_recording = []
    loopback_recording = []

    record_samples = int(sample_rate * calibration_duration_sec)
    stop_recording = threading.Event()

    def record_mic_loop():
        try:
            blocksize = int(sample_rate * 60 / 1000)
            with mic.recorder(samplerate=sample_rate, blocksize=blocksize) as r:
                chunk_size = int(sample_rate * 0.05)
                recorded = 0
                while recorded < record_samples and not stop_recording.is_set():
                    data = r.record(numframes=chunk_size)
                    mono = data.mean(axis=1) if data.ndim > 1 else data
                    mic_recording.append(mono.astype(np.float32))
                    recorded += len(mono)
        except Exception as e:
            print(f"[AEC-Calibration] Mic recording error: {e}")

    def record_loopback_loop():
        try:
            blocksize = int(sample_rate * 60 / 1000)
            with loopback_mic.recorder(samplerate=sample_rate, blocksize=blocksize) as r:
                chunk_size = int(sample_rate * 0.05)
                recorded = 0
                while recorded < record_samples and not stop_recording.is_set():
                    data = r.record(numframes=chunk_size)
                    mono = data.mean(axis=1) if data.ndim > 1 else data
                    loopback_recording.append(mono.astype(np.float32))
                    recorded += len(mono)
        except Exception as e:
            print(f"[AEC-Calibration] Loopback recording error: {e}")

    def play_chirp_loop():
        try:
            time.sleep(0.05)  # Let recording warm up
            try:
                with speaker.player(samplerate=sample_rate, channels=1) as p:
                    p.play(play_signal)
            except Exception:
                # Fallback to stereo playback if mono player fails
                stereo_signal = np.column_stack((play_signal, play_signal))
                with speaker.player(samplerate=sample_rate, channels=2) as p:
                    p.play(stereo_signal)
        except Exception as e:
            print(f"[AEC-Calibration] Playback error: {e}")

    # Start threads
    t_mic = threading.Thread(target=record_mic_loop, daemon=True)
    t_loop = threading.Thread(target=record_loopback_loop, daemon=True)
    t_play = threading.Thread(target=play_chirp_loop, daemon=True)

    t_mic.start()
    t_loop.start()
    t_play.start()

    # Wait for the recording window to complete
    time.sleep(calibration_duration_sec + 0.1)
    stop_recording.set()

    t_mic.join(timeout=0.5)
    t_loop.join(timeout=0.5)
    t_play.join(timeout=0.5)

    if not mic_recording or not loopback_recording:
        print("[AEC-Calibration] Warning: Calibration failed to capture audio. Using default 80ms delay.")
        return 80

    mic_audio = np.concatenate(mic_recording)[:record_samples]
    loopback_audio = np.concatenate(loopback_recording)[:record_samples]

    # Compute cross-correlation
    corr = np.correlate(mic_audio, loopback_audio, mode='full')
    zero_lag = len(loopback_audio) - 1

    # Search for positive lags (mic captures after loopback) up to 300ms (4800 samples @ 16kHz)
    max_search_samples = int(sample_rate * 0.3)
    lags_to_check = corr[zero_lag : zero_lag + max_search_samples]

    if len(lags_to_check) == 0:
        print("[AEC-Calibration] Warning: Correlation buffer empty. Using default 80ms delay.")
        return 80

    best_lag = int(np.argmax(lags_to_check))
    peak_val = lags_to_check[best_lag]

    # Compute normalized correlation coefficient at the best lag to check signal quality
    # r = sum(mic * ref) / sqrt(sum(mic^2) * sum(ref^2))
    mic_slice = mic_audio[best_lag:]
    loopback_slice = loopback_audio[:len(mic_slice)]

    mic_energy = np.sum(mic_slice**2)
    loopback_energy = np.sum(loopback_slice**2)
    den = (mic_energy * loopback_energy)**0.5

    norm_corr = float(peak_val / den) if den > 1e-9 else 0.0
    measured_delay_ms = int((best_lag / sample_rate) * 1000)

    print(f"[AEC-Calibration] Measured delay: {measured_delay_ms} ms (best lag: {best_lag} samples, normalized correlation: {norm_corr:.4f})")

    # Accept calibration if correlation is above 0.15
    if norm_corr >= 0.15:
        print(f"[AEC-Calibration] Calibration successful! Dynamic delay set to {measured_delay_ms} ms.")
        return measured_delay_ms
    else:
        print(f"[AEC-Calibration] Weak acoustic path detected (correlation {norm_corr:.4f} < 0.15). "
              f"Either headphones are in use or speakers are muted. Falling back to default 80ms delay.")
        return 80

