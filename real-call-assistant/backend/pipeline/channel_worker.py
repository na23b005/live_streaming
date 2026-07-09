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

from .transcript_normalizer import clean_text

# Global lock to serialize GPU/CPU inference across channels,
# preventing concurrent execution deadlocks in ONNX Runtime/DirectML.
stt_lock = threading.Lock()


@dataclass
class TranscriptEvent:
    speaker: str
    start_ts: float
    end_ts: float
    text: str


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
    ):
        self.speaker_label = speaker_label
        self.capture = capture
        self.segmenter = segmenter
        self.engine = engine
        self.out_queue = out_queue
        self.echo_canceller = echo_canceller
        self.shared_state = shared_state if shared_state is not None else {}
        self.default_rms_threshold = getattr(segmenter, "rms_threshold", 0.008)
        self.processed_queue = queue.Queue()
        self.audio_history = []
        self._segments: "queue.Queue[SpeechSegment]" = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        self.capture.start()
        preprocess_thread = threading.Thread(
            target=self._preprocess_audio_loop,
            daemon=True,
        )
        vad_thread = threading.Thread(
            target=self.segmenter.run,
            args=(self.processed_queue, self._segments, self._stop),
            daemon=True,
        )
        stt_thread = threading.Thread(target=self._stt_loop, daemon=True)
        preprocess_thread.start()
        vad_thread.start()
        stt_thread.start()
        self._threads = [preprocess_thread, vad_thread, stt_thread]

    def _preprocess_audio_loop(self) -> None:
        import time
        while not self._stop.is_set():
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
                    cleaned_chunk = self.echo_canceller.process_mic(chunk, timestamp)
                    self.audio_history.append(cleaned_chunk)
                    self.processed_queue.put((timestamp, cleaned_chunk))
                else:
                    # Loopback channel: push to reference buffer and pass through unchanged
                    self.echo_canceller.push_reference(chunk, timestamp)
                    self.audio_history.append(chunk)
                    self.processed_queue.put((timestamp, chunk))
            else:
                self.audio_history.append(chunk)
                self.processed_queue.put((timestamp, chunk))

    def stop(self) -> None:
        self._stop.set()
        self.capture.stop()
        for t in self._threads:
            t.join(timeout=2)

    def _stt_loop(self) -> None:
        while not self._stop.is_set() or not self._segments.empty():
            try:
                timeout = 0.5 if not self._stop.is_set() else 0.05
                segment = self._segments.get(timeout=timeout)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            
            # Calculate segment average RMS energy
            rms = np.sqrt(np.mean(segment.audio**2)) if len(segment.audio) > 0 else 0.0
            
            # Skip extremely quiet microphone segments to avoid transcribing noise
            if self.speaker_label == "Speaker 1" and rms < 0.012:
                print(f"Skipping segment: quiet mic audio (RMS: {rms:.4f})")
                continue

            start_t = time.perf_counter()
            with stt_lock:
                text = clean_text(self.engine.transcribe(segment.audio, self.capture.samplerate))
            completed_at = time.perf_counter()
            
            if not text:
                continue
                
            # Filter common Whisper hallucinations on low energy for microphone channel
            if self.speaker_label == "Speaker 1":
                cleaned_lower = text.lower().strip().replace("’", "'").translate(str.maketrans("", "", ".,?!"))
                if cleaned_lower in ["thank you", "thank you so much", "i don't know", "you", "yeah", "yes", "oh", "bye"]:
                    if rms < 0.025:
                        print(f"Skipping segment: suspected Whisper hallucination '{text}' on low energy (RMS: {rms:.4f})")
                        continue
                
            # Compute latency metrics
            stt_duration = completed_at - start_t
            
            # The hangover duration is the silence window the local VAD waits for to confirm speech ended
            hangover_s = self.segmenter.hangover_frames * (self.segmenter.frame_samples / self.segmenter.samplerate)
            
            # The physical speaking actually ended hangover_s seconds before segment was created/cut
            physical_ended_at = segment.created_at - hangover_s
            total_latency = completed_at - physical_ended_at
            
            print(f"\n--- Latency Report for [{self.speaker_label}] ---")
            print(f"Text: \"{text}\"")
            print(f" ➜ local VAD Hangover:   {hangover_s*1000:.0f} ms  (time spent waiting to confirm silence)")
            print(f" ➜ Network RTT & GPU STT: {stt_duration*1000:.0f} ms  (Tailscale + RTX 5090 compilation/inference)")
            print(f" ➜ Total E2E Latency:     {total_latency*1000:.0f} ms  (from end of speaking to transcript display)")
            print("-" * 40 + "\n")
            
            self.out_queue.put(
                TranscriptEvent(
                    speaker=self.speaker_label,
                    start_ts=segment.start_ts,
                    end_ts=segment.end_ts,
                    text=text,
                )
            )
