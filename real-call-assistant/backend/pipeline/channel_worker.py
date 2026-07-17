"""
Wires one audio channel (mic OR system loopback) end-to-end:

    AudioCapture -> VADSegmenter -> STTEngine -> TranscriptEvent

Each stage runs on its own thread, so a slow transcription on one channel
never blocks audio capture on either channel. Two ChannelWorker instances
(one per channel) push onto the same shared output queue, which is how the
two speakers ("Me" and "Speaker 1") end up interleaved in one live transcript.
"""

import queue
import time
import threading
import numpy as np
from dataclasses import dataclass

from audio.capture import AudioCapture
from audio.vad_segmenter import SpeechSegment, VADSegmenter
from stt.base import STTEngine

from .transcript_normalizer import clean_text, is_whisper_hallucination, longest_common_prefix, remove_boundary_overlap

# Global lock to serialize GPU/CPU inference across channels,
# preventing concurrent execution deadlocks in ONNX Runtime/DirectML.
stt_lock = threading.Lock()


class STTCircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_timeout=60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = 0.0

    def can_transcribe(self) -> bool:
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                print("[CircuitBreaker] Entering HALF_OPEN state - testing GPU/STT recovery...")
                return True
            return False
        return True

    def record_success(self):
        if self.state != "CLOSED":
            print("[CircuitBreaker] Transaction succeeded. Closing circuit (GPU/STT recovered).")
        self.failures = 0
        self.state = "CLOSED"

    def record_failure(self) -> bool:
        """Returns True if the circuit just tripped from CLOSED/HALF_OPEN to OPEN."""
        self.failures += 1
        self.last_failure_time = time.time()
        print(f"[CircuitBreaker] Recorded failure {self.failures}/{self.failure_threshold}.")
        if self.state != "OPEN" and self.failures >= self.failure_threshold:
            self.state = "OPEN"
            print(f"[CircuitBreaker] State changed to OPEN. STT disabled for the next {self.recovery_timeout} seconds.")
            return True
        return False


stt_circuit_breaker = STTCircuitBreaker()


@dataclass
class TranscriptEvent:
    speaker: str
    start_ts: float
    end_ts: float
    text: str
    is_final: bool = True


class ChannelWorker:
    def __init__(
        self,
        speaker_label: str,
        capture: AudioCapture,
        segmenter: VADSegmenter,
        engine: STTEngine,
        out_queue: "queue.Queue[TranscriptEvent]",
        echo_canceller=None,
        shared_state: dict = None,
        meeting_id: str | None = None,
    ):
        self.speaker_label = speaker_label
        self.capture = capture
        self.segmenter = segmenter
        self.engine = engine
        self.out_queue = out_queue
        self.echo_canceller = echo_canceller
        self.shared_state = shared_state if shared_state is not None else {}
        self.meeting_id = meeting_id
        self.default_rms_threshold = getattr(segmenter, "rms_threshold", 0.008)
        self.last_final_text = ""
        self.last_final_ts = 0.0
        self.processed_queue = queue.Queue(maxsize=100)
        self.audio_history = []
        self._segments: "queue.Queue[SpeechSegment]" = queue.Queue(maxsize=100)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._thread_health = {
            'preprocess': {'last_ping': time.time(), 'alive': False},
            'vad': {'last_ping': time.time(), 'alive': False},
            'stt': {'last_ping': time.time(), 'alive': False},
            'speculative': {'last_ping': time.time(), 'alive': False},
        }
        self._restart_counts = {
            'preprocess': 0,
            'vad': 0,
            'stt': 0,
            'speculative': 0
        }
        self._max_restarts = 3
        self._supervisor: threading.Thread | None = None

    def start(self) -> None:
        self.capture.start()
        
        now = time.time()
        for name in self._thread_health:
            self._thread_health[name]['last_ping'] = now
            self._thread_health[name]['alive'] = True

        preprocess_thread = threading.Thread(
            target=self._preprocess_audio_loop,
            name=f"{self.speaker_label}_preprocess",
            daemon=True,
        )
        vad_thread = threading.Thread(
            target=self.segmenter.run,
            args=(self.processed_queue, self._segments, self._stop),
            kwargs={'ping_callback': lambda: self._ping_thread('vad')},
            name=f"{self.speaker_label}_vad",
            daemon=True,
        )
        stt_thread = threading.Thread(
            target=self._stt_loop,
            name=f"{self.speaker_label}_stt",
            daemon=True
        )
        speculative_thread = threading.Thread(
            target=self._speculative_stt_loop,
            name=f"{self.speaker_label}_speculative",
            daemon=True
        )
        
        preprocess_thread.start()
        vad_thread.start()
        stt_thread.start()
        speculative_thread.start()
        self._threads = [preprocess_thread, vad_thread, stt_thread, speculative_thread]
        
        self._supervisor = threading.Thread(target=self._supervise, daemon=True)
        self._supervisor.start()

    def _ping_thread(self, name: str) -> None:
        if name in self._thread_health:
            self._thread_health[name]['last_ping'] = time.time()

    def _put_queue(self, q, item, timeout=2.0) -> None:
        try:
            q.put(item, timeout=timeout)
        except queue.Full:
            try:
                q.get_nowait()
                q.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass

    def _preprocess_audio_loop(self) -> None:
        import time
        while not self._stop.is_set():
            self._ping_thread('preprocess')
            try:
                item = self.capture.out_queue.get(timeout=0.5)
                timestamp, chunk = item
            except queue.Empty:
                continue

            # Update shared state if loopback channel has active audio
            if self.speaker_label == "Speaker 2":
                rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0.0
                if rms > 0.01:
                    self.shared_state["last_sys_audio_active_time"] = time.time()
            elif self.speaker_label == "Speaker 1":
                # Dynamically raise mic VAD RMS threshold if loopback was active recently (room echo decay masking)
                time_since_sys = time.time() - self.shared_state.get("last_sys_audio_active_time", 0.0)
                if time_since_sys < 1.0:
                    self.segmenter.rms_threshold = 0.035
                else:
                    self.segmenter.rms_threshold = self.default_rms_threshold

            if self.echo_canceller:
                if self.speaker_label == "Speaker 1":
                    # Microphone channel: cancel echo using the aligned reference audio
                    cleaned_chunk, proc_ts = self.echo_canceller.process_mic(chunk, timestamp)
                    if len(cleaned_chunk) > 0:
                        self.audio_history.append(cleaned_chunk)
                        self._put_queue(self.processed_queue, (proc_ts, cleaned_chunk))
                else:
                    # Loopback channel: push to reference buffer and pass through unchanged
                    self.echo_canceller.push_reference(chunk, timestamp)
                    self.audio_history.append(chunk)
                    self._put_queue(self.processed_queue, (timestamp, chunk))
            else:
                self.audio_history.append(chunk)
                self._put_queue(self.processed_queue, (timestamp, chunk))

    def stop(self) -> None:
        self._stop.set()
        self.capture.stop()
        for t in self._threads:
            t.join(timeout=2)

    def _stt_loop(self) -> None:
        while not self._stop.is_set() or not self._segments.empty():
            self._ping_thread('stt')
            try:
                timeout = 0.5 if not self._stop.is_set() else 0.05
                segment = self._segments.get(timeout=timeout)
                self.last_final_ts = segment.start_ts
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            
            # Calculate segment average RMS energy
            rms = np.sqrt(np.mean(segment.audio**2)) if len(segment.audio) > 0 else 0.0
            
            # Skip extremely quiet segments to avoid transcribing noise/silence
            rms_threshold = 0.005 if self.speaker_label == "Speaker 1" else 0.003
            if rms < rms_threshold:
                print(f"Skipping segment: quiet {self.speaker_label} audio (RMS: {rms:.4f})")
                self._put_queue(
                    self.out_queue,
                    TranscriptEvent(
                        speaker=self.speaker_label,
                        start_ts=segment.start_ts,
                        end_ts=segment.end_ts,
                        text="",
                        is_final=True
                    )
                )
                continue

            # Guard 1: Segment-Level Correlation and RMS ratio check
            if self.speaker_label == "Speaker 1" and self.echo_canceller:
                abs_start = segment.created_at - (segment.end_ts - segment.start_ts)
                corr, ref_rms = self.echo_canceller.measure_segment_correlation_and_rms(segment.audio, abs_start)
                volume_ratio = rms / ref_rms if ref_rms > 1e-6 else 999.0

                # Print stats for diagnostics (requested by user)
                print(f"[AEC-Gate] Segment diagnostics -> Correlation: {corr:.4f}, Mic RMS: {rms:.4f}, Speaker RMS: {ref_rms:.4f}, Volume Ratio: {volume_ratio:.2f}")

                # If correlation is high (highly similar to speaker) and volume ratio is low-to-moderate
                if corr > 0.22 and volume_ratio < 1.5:
                    print(f"[AEC-Gate] Vetoed/Discarded segment on {self.speaker_label} due to high loopback correlation ({corr:.4f}) and low/equal volume ratio ({volume_ratio:.2f})")
                    self._put_queue(
                        self.out_queue,
                        TranscriptEvent(
                            speaker=self.speaker_label,
                            start_ts=segment.start_ts,
                            end_ts=segment.end_ts,
                            text="",
                            is_final=True
                        )
                    )
                    continue

            # Run transcription with lock timeout and circuit breaker protection
            if not stt_circuit_breaker.can_transcribe():
                print(f"[CircuitBreaker] Skipping transcription for segment on {self.speaker_label} (circuit is OPEN).")
                self._put_queue(
                    self.out_queue,
                    TranscriptEvent(
                        speaker=self.speaker_label,
                        start_ts=segment.start_ts,
                        end_ts=segment.end_ts,
                        text="",
                        is_final=True
                    )
                )
                continue

            start_t = time.perf_counter()
            acquired = stt_lock.acquire(timeout=30.0)
            if not acquired:
                print(f"CRITICAL: STT lock acquisition timed out for {self.speaker_label} — GPU thread stuck")
                tripped = stt_circuit_breaker.record_failure()
                if tripped:
                    self._put_queue(
                        self.out_queue,
                        TranscriptEvent(
                            speaker="System Warning",
                            start_ts=0.0,
                            end_ts=0.0,
                            text="[CRITICAL] STT engine circuit breaker is OPEN. Transcription is temporarily disabled due to consecutive GPU/Inference failures."
                        )
                    )
                self._put_queue(
                    self.out_queue,
                    TranscriptEvent(
                        speaker=self.speaker_label,
                        start_ts=segment.start_ts,
                        end_ts=segment.end_ts,
                        text="",
                        is_final=True
                    )
                )
                continue

            try:
                text = clean_text(self.engine.transcribe(segment.audio, self.capture.samplerate, self.meeting_id))
                stt_circuit_breaker.record_success()
            except Exception as e:
                print(f"Error during transcription on {self.speaker_label}: {e}")
                tripped = stt_circuit_breaker.record_failure()
                if tripped:
                    self._put_queue(
                        self.out_queue,
                        TranscriptEvent(
                            speaker="System Warning",
                            start_ts=0.0,
                            end_ts=0.0,
                            text="[CRITICAL] STT engine circuit breaker is OPEN. Transcription is temporarily disabled due to consecutive GPU/Inference failures."
                        )
                    )
                self._put_queue(
                    self.out_queue,
                    TranscriptEvent(
                        speaker=self.speaker_label,
                        start_ts=segment.start_ts,
                        end_ts=segment.end_ts,
                        text="",
                        is_final=True
                    )
                )
                continue
            finally:
                stt_lock.release()

            completed_at = time.perf_counter()
            
            if not text:
                self._put_queue(
                    self.out_queue,
                    TranscriptEvent(
                        speaker=self.speaker_label,
                        start_ts=segment.start_ts,
                        end_ts=segment.end_ts,
                        text="",
                        is_final=True
                    )
                )
                continue
                
            # Filter common Whisper hallucinations on low energy
            if is_whisper_hallucination(text, rms):
                print(f"Skipping segment: suspected Whisper hallucination '{text}' (RMS: {rms:.4f}, channel: {self.speaker_label})")
                self._put_queue(
                    self.out_queue,
                    TranscriptEvent(
                        speaker=self.speaker_label,
                        start_ts=segment.start_ts,
                        end_ts=segment.end_ts,
                        text="",
                        is_final=True
                    )
                )
                continue
                
            # Deduplicate overlap at boundaries
            deduplicated_text = remove_boundary_overlap(self.last_final_text, text)
            self.last_final_text = text

            if not deduplicated_text.strip():
                print(f"Skipping segment on {self.speaker_label}: completely duplicate text after boundary deduplication")
                self._put_queue(
                    self.out_queue,
                    TranscriptEvent(
                        speaker=self.speaker_label,
                        start_ts=segment.start_ts,
                        end_ts=segment.end_ts,
                        text="",
                        is_final=True
                    )
                )
                continue

            # Compute latency metrics
            stt_duration = completed_at - start_t

            # The hangover duration is the silence window the local VAD waits for to confirm speech ended
            hangover_s = self.segmenter.hangover_frames * (self.segmenter.frame_samples / self.segmenter.samplerate)

            # The physical speaking actually ended hangover_s seconds before segment was created/cut
            physical_ended_at = segment.created_at - hangover_s
            total_latency = completed_at - physical_ended_at

            print(f"\n--- Latency Report for [{self.speaker_label}] ---")
            print(f"Text: \"{deduplicated_text}\" (original: \"{text}\")")
            print(f" ➜ local VAD Hangover:   {hangover_s*1000:.0f} ms  (time spent waiting to confirm silence)")
            print(f" ➜ Network RTT & GPU STT: {stt_duration*1000:.0f} ms  (Tailscale + RTX 5090 compilation/inference)")
            print(f" ➜ Total E2E Latency:     {total_latency*1000:.0f} ms  (from end of speaking to transcript display)")
            print("-" * 40 + "\n")

            self._put_queue(
                self.out_queue,
                TranscriptEvent(
                    speaker=self.speaker_label,
                    start_ts=segment.start_ts,
                    end_ts=segment.end_ts,
                    text=deduplicated_text,
                )
            )
        
    def _speculative_stt_loop(self) -> None:
        last_raw_text = ""
        last_emitted_text = ""
        last_speculative_audio_len = 0
        last_start_ts = 0.0
        last_end_ts = 0.0
        
        while not self._stop.is_set():
            time.sleep(1.5)
            self._ping_thread('speculative')
            
            # Check circuit breaker
            if not stt_circuit_breaker.can_transcribe():
                continue
                
            # Peek active speech segment
            segment = self.segmenter.peek_active_segment()
            if segment is None:
                # If there's no active speech but we have pending speculative text, flush it fully
                if last_raw_text and last_raw_text != last_emitted_text and last_start_ts > self.last_final_ts:
                    self._put_queue(
                        self.out_queue,
                        TranscriptEvent(
                            speaker=self.speaker_label,
                            start_ts=last_start_ts,
                            end_ts=last_end_ts,
                            text=last_raw_text,
                            is_final=False
                        )
                    )
                # Reset speculation state
                last_raw_text = ""
                last_emitted_text = ""
                last_speculative_audio_len = 0
                last_start_ts = 0.0
                last_end_ts = 0.0
                continue
                
            # Skip if we already processed this exact audio length (or extremely close) to save CPU/GPU cycles
            if abs(len(segment.audio) - last_speculative_audio_len) < 16000 * 0.2:
                continue
                
            last_speculative_audio_len = len(segment.audio)
            
            # Acquire STT lock
            acquired = stt_lock.acquire(timeout=5.0) # short timeout to avoid delaying completed final segments!
            if not acquired:
                continue
                
            try:
                # Transcribe the active growing segment copy
                text = clean_text(self.engine.transcribe(segment.audio, self.capture.samplerate, self.meeting_id))
                
                if not text:
                    continue
                    
                # Filter common Whisper hallucinations on low energy
                rms = np.sqrt(np.mean(segment.audio**2)) if len(segment.audio) > 0 else 0.0
                if is_whisper_hallucination(text, rms):
                    continue
                    
                # LocalAgreement-2 Algorithm:
                # First pass: seed the last raw text and wait for consensus
                if not last_raw_text:
                    last_raw_text = text
                    last_start_ts = segment.start_ts
                    last_end_ts = segment.end_ts
                    continue
                    
                # Find the Longest Common Prefix (LCP) between the previous pass and this pass
                agreed_text = longest_common_prefix(last_raw_text, text)
                last_raw_text = text
                last_start_ts = segment.start_ts
                last_end_ts = segment.end_ts
                
                if agreed_text:
                    # Strip to last word boundary to avoid cutting off words mid-stream
                    last_space_idx = agreed_text.rfind(" ")
                    if last_space_idx != -1:
                        agreed_text = agreed_text[:last_space_idx].strip()
                        
                    if len(agreed_text) > len(last_emitted_text):
                        last_emitted_text = agreed_text
                        
                        # Emit speculative partial event
                        self._put_queue(
                            self.out_queue,
                            TranscriptEvent(
                                speaker=self.speaker_label,
                                start_ts=segment.start_ts,
                                end_ts=segment.end_ts,
                                text=agreed_text,
                                is_final=False
                            )
                        )
            except Exception as e:
                print(f"Error during speculative transcription: {e}")
            finally:
                stt_lock.release()

    def _supervise(self) -> None:
        while not self._stop.is_set():
            time.sleep(5)
            if self._stop.is_set():
                break

            now = time.time()
            for name, health in self._thread_health.items():
                if not health['alive']:
                    continue

                # Locate corresponding thread in self._threads
                expected_name = f"{self.speaker_label}_{name}"
                thread_obj = None
                for t in self._threads:
                    if t and t.name == expected_name:
                        thread_obj = t
                        break

                is_thread_dead = thread_obj is None or not thread_obj.is_alive()
                is_thread_hung = (now - health['last_ping']) > 15.0

                if is_thread_dead or is_thread_hung:
                    reason = "died" if is_thread_dead else "hung (no pings)"
                    print(f"FATAL: STT pipeline thread '{name}' in {self.speaker_label} {reason}.")

                    # Broadcast error to UI
                    self._put_queue(
                        self.out_queue,
                        TranscriptEvent(
                            speaker="System Warning",
                            start_ts=0.0,
                            end_ts=0.0,
                            text=f"[CRITICAL] STT pipeline thread '{name}' on channel '{self.speaker_label}' {reason}. Attempting recovery..."
                        )
                    )

                    # Check restart count
                    if self._restart_counts[name] < self._max_restarts:
                        self._restart_counts[name] += 1
                        print(f"[Supervisor] Attempting restart {self._restart_counts[name]}/{self._max_restarts} for thread '{name}'...")
                        
                        # Reset health metrics before starting
                        health['last_ping'] = time.time()
                        health['alive'] = True

                        if name == 'preprocess':
                            new_t = threading.Thread(
                                target=self._preprocess_audio_loop,
                                name=expected_name,
                                daemon=True
                            )
                        elif name == 'vad':
                            # Reset segmenter states to avoid corrupted segmentation on restart
                            try:
                                self.segmenter.vad.reset()
                            except Exception:
                                pass
                            new_t = threading.Thread(
                                target=self.segmenter.run,
                                args=(self.processed_queue, self._segments, self._stop),
                                kwargs={'ping_callback': lambda: self._ping_thread('vad')},
                                name=expected_name,
                                daemon=True
                            )
                        elif name == 'stt':
                            new_t = threading.Thread(
                                target=self._stt_loop,
                                name=expected_name,
                                daemon=True
                            )

                        # Replace in thread list and start
                        if thread_obj in self._threads:
                            self._threads.remove(thread_obj)
                        new_t.start()
                        self._threads.append(new_t)
                    else:
                        print(f"[Supervisor] Restart limit exceeded for thread '{name}' on channel '{self.speaker_label}'. Shutting down worker.")
                        health['alive'] = False
                        
                        # Trigger graceful worker shutdown
                        self._put_queue(
                            self.out_queue,
                            TranscriptEvent(
                                speaker="System Warning",
                                start_ts=0.0,
                                end_ts=0.0,
                                text=f"[CRITICAL] Restart limit exceeded for STT pipeline thread '{name}' on channel '{self.speaker_label}'. Recording is stopping."
                            )
                        )
                        self._stop.set()
                        break
