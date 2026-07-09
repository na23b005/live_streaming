import threading
import time
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
        ref_threshold: float = 0.01
    ):
        self.sample_rate = sample_rate
        self.delay_sec = delay_ms / 1000.0
        self.ducking_threshold = ducking_threshold
        self.ref_threshold = ref_threshold
        
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
        
        # Keep maximum of 5 seconds of reference history
        self.max_history_samples = 5 * self.sample_rate
        # Prune when it exceeds 8 seconds, leaving last 3 seconds
        self.prune_threshold_samples = 8 * self.sample_rate
        self.prune_to_samples = 3 * self.sample_rate

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

    def process_mic(self, mic_chunk: np.ndarray, mic_timestamp: float) -> np.ndarray:
        """Aligns the mic chunk with reference history and applies WebRTC AEC and ducking."""
        # Find corresponding reference audio from history
        with self.lock:
            if self.ref_start_time is None or len(self.all_ref_samples) == 0:
                # No reference audio captured yet; return original mic chunk
                return mic_chunk.copy()
            
            target_time = mic_timestamp - self.delay_sec
            start_idx = int((target_time - self.ref_start_time) * self.sample_rate)
            chunk_len = len(mic_chunk)
            
            # Extract aligned reference chunk, padding with zeros where reference data is missing
            if start_idx < 0:
                # Part of the target window is before the start of the reference buffer
                missing_prefix_samples = -start_idx
                if missing_prefix_samples >= chunk_len:
                    ref_chunk = np.zeros(chunk_len, dtype=np.float32)
                else:
                    available = self.all_ref_samples[0 : chunk_len - missing_prefix_samples]
                    ref_chunk = np.concatenate((np.zeros(missing_prefix_samples, dtype=np.float32), available))
            else:
                end_idx = start_idx + chunk_len
                if end_idx <= len(self.all_ref_samples):
                    ref_chunk = self.all_ref_samples[start_idx:end_idx].copy()
                else:
                    # Part of the target window is not yet captured in the reference buffer
                    available_samples = len(self.all_ref_samples) - start_idx
                    if available_samples <= 0:
                        ref_chunk = np.zeros(chunk_len, dtype=np.float32)
                    else:
                        available = self.all_ref_samples[start_idx:]
                        missing_suffix_samples = chunk_len - len(available)
                        ref_chunk = np.concatenate((available, np.zeros(missing_suffix_samples, dtype=np.float32)))

            # --- Cross-Channel Energy Ducking / Echo Suppression Gate ---
            mic_rms = np.sqrt(np.mean(mic_chunk**2)) if len(mic_chunk) > 0 else 0.0
            max_ref_rms = max(self.ref_rms_history) if self.ref_rms_history else 0.0
            
            # If system speaker is active
            if max_ref_rms > self.ref_threshold:
                ratio = mic_rms / max_ref_rms
                # If mic energy is low relative to system speaker (meaning it is pure echo leakage)
                # AND mic energy is below an absolute active threshold (prevent muting loud user speech)
                if ratio < self.ducking_threshold and mic_rms < 0.05:
                    # Suppress the echo completely (setting to zero tells VAD it is silent)
                    return np.zeros_like(mic_chunk)

        # Run WebRTC AEC in 10ms frames
        num_samples = len(mic_chunk)
        clean_samples = []
        
        # Convert float32 arrays (-1.0 to 1.0) to PCM16 signed integers
        mic_pcm = (np.clip(mic_chunk, -1.0, 1.0) * 32767).astype(np.int16)
        ref_pcm = (np.clip(ref_chunk, -1.0, 1.0) * 32767).astype(np.int16)
        
        for i in range(0, num_samples, self.frame_samples):
            mic_frame = mic_pcm[i : i + self.frame_samples]
            ref_frame = ref_pcm[i : i + self.frame_samples]
            
            # Pad frames to 10ms size if needed (e.g. trailing samples of last chunk)
            if len(mic_frame) < self.frame_samples:
                mic_frame = np.pad(mic_frame, (0, self.frame_samples - len(mic_frame)))
            if len(ref_frame) < self.frame_samples:
                ref_frame = np.pad(ref_frame, (0, self.frame_samples - len(ref_frame)))
            
            # Feed reference (loopback) to reverse stream
            self.ap.process_reverse_stream(ref_frame.tobytes())
            
            # Cancel echo in microphone stream
            clean_frame_bytes = self.ap.process_stream(mic_frame.tobytes())
            
            # Convert back to float32
            clean_frame = np.frombuffer(clean_frame_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            
            # Truncate if we padded it and it was the last chunk
            actual_len = len(mic_chunk) - i
            if actual_len < self.frame_samples:
                clean_frame = clean_frame[:actual_len]
                
            clean_samples.append(clean_frame)
            
        return np.concatenate(clean_samples)
