"""
Wires one audio channel (mic OR system loopback) end-to-end:

    AudioCapture -> VADSegmenter -> STTEngine -> TranscriptEvent

Each stage runs on its own thread, so a slow transcription on one channel
never blocks audio capture on either channel. Two ChannelWorker instances
(one per channel) push onto the same shared output queue, which is how the
two speakers ("Me" and "Speaker 1") end up interleaved in one live transcript.
"""

import queue
import threading
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
    ):
        self.speaker_label = speaker_label
        self.capture = capture
        self.segmenter = segmenter
        self.engine = engine
        self.out_queue = out_queue
        self._segments: "queue.Queue[SpeechSegment]" = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        self.capture.start()
        vad_thread = threading.Thread(
            target=self.segmenter.run,
            args=(self.capture.out_queue, self._segments, self._stop),
            daemon=True,
        )
        stt_thread = threading.Thread(target=self._stt_loop, daemon=True)
        vad_thread.start()
        stt_thread.start()
        self._threads = [vad_thread, stt_thread]

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
            with stt_lock:
                text = clean_text(self.engine.transcribe(segment.audio, self.capture.samplerate))
            if not text:
                continue
            self.out_queue.put(
                TranscriptEvent(
                    speaker=self.speaker_label,
                    start_ts=segment.start_ts,
                    end_ts=segment.end_ts,
                    text=text,
                )
            )
