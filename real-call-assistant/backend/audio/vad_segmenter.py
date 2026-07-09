"""
Turns a stream of small audio chunks into discrete speech segments using
WebRTC VAD, and hands each finished segment to Whisper.

This replaces Natively's LocalAgreement-2 streaming logic with something
simpler: instead of re-running inference every ~1.5s on a sliding window and
keeping only the agreed-upon prefix (which gives interim/partial results),
we just wait for a natural pause in speech (silence_hangover_ms) and
transcribe the whole utterance once. Simpler to implement correctly, at the
cost of not showing partial/interim text while someone is still talking.
If you want live word-by-word partials later, LocalAgreement-2 (or a
streaming-native model like Moonshine) is the upgrade path - see README.
"""

import queue
import time
from dataclasses import dataclass

import numpy as np
import webrtcvad


@dataclass
class SpeechSegment:
    audio: np.ndarray   # float32 mono, at the configured sample rate
    start_ts: float      # seconds since this segmenter started running
    end_ts: float
    created_at: float = 0.0


class VADSegmenter:
    def __init__(
        self,
        samplerate: int,
        frame_ms: int,
        aggressiveness: int,
        silence_hangover_ms: int,
        min_speech_ms: int,
        max_segment_s: float,
        rms_threshold: float = 0.0,
    ):
        self.samplerate = samplerate
        self.frame_samples = int(samplerate * frame_ms / 1000)
        self.vad = webrtcvad.Vad(aggressiveness)
        self.hangover_frames = max(1, silence_hangover_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.max_segment_frames = int(max_segment_s * 1000 / frame_ms)
        self.rms_threshold = rms_threshold

    @staticmethod
    def _to_pcm16(chunk: np.ndarray) -> bytes:
        clipped = np.clip(chunk, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16).tobytes()

    def run(
        self,
        in_queue: "queue.Queue[np.ndarray]",
        out_queue: "queue.Queue[SpeechSegment]",
        stop_event,
    ) -> None:
        t0 = time.monotonic()
        speech_frames: list[np.ndarray] = []
        silence_run = 0
        speech_start_ts: float | None = None

        while not stop_event.is_set():
            try:
                item = in_queue.get(timeout=0.5)
                if isinstance(item, tuple):
                    timestamp, chunk = item
                else:
                    timestamp, chunk = time.perf_counter(), item
            except queue.Empty:
                continue

            # webrtcvad requires exact frame_samples-length int16 frames.
            if len(chunk) < self.frame_samples:
                chunk = np.pad(chunk, (0, self.frame_samples - len(chunk)))
            elif len(chunk) > self.frame_samples:
                chunk = chunk[: self.frame_samples]

            # WebRTC VAD check
            is_speech = self.vad.is_speech(self._to_pcm16(chunk), self.samplerate)
            
            # Absolute energy gate: if frame energy is below self.rms_threshold, treat as silence.
            # This filters out low-level microphone hum/static that fools WebRTC VAD.
            if self.rms_threshold > 0.0:
                rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0.0
                if rms < self.rms_threshold:
                    is_speech = False

            now_ts = time.monotonic() - t0

            if is_speech:
                if not speech_frames:
                    speech_start_ts = now_ts
                speech_frames.append(chunk)
                silence_run = 0
            elif speech_frames:
                # Keep trailing silence in the buffer for a natural cut-off,
                # but count it toward the hangover threshold.
                speech_frames.append(chunk)
                silence_run += 1

            hit_hangover = speech_frames and silence_run >= self.hangover_frames
            # Cut at a natural pause after target length, or unconditionally at 1.5x target length
            hit_max_len = (len(speech_frames) >= self.max_segment_frames and not is_speech) or (len(speech_frames) >= int(self.max_segment_frames * 1.5))
            if speech_frames and (hit_hangover or hit_max_len):
                if len(speech_frames) - silence_run >= self.min_speech_frames:
                    audio = np.concatenate(speech_frames).astype(np.float32)
                    out_queue.put(
                        SpeechSegment(
                            audio=audio,
                            start_ts=speech_start_ts,
                            end_ts=now_ts,
                            created_at=time.perf_counter()
                        )
                    )
                speech_frames = []
                silence_run = 0
                speech_start_ts = None

        # Flush any remaining active speech frames in the buffer upon shutdown
        if speech_frames:
            actual_speech_len = len(speech_frames) - silence_run
            if actual_speech_len >= self.min_speech_frames:
                audio = np.concatenate(speech_frames).astype(np.float32)
                now_ts = time.monotonic() - t0
                out_queue.put(
                    SpeechSegment(
                        audio=audio,
                        start_ts=speech_start_ts,
                        end_ts=now_ts,
                        created_at=time.perf_counter()
                    )
                )
