"""
Turns a stream of small audio chunks into discrete speech segments using
Silero VAD (ONNX), and hands each finished segment to Whisper.

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
import os
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort


@dataclass
class SpeechSegment:
    audio: np.ndarray   # float32 mono, at the configured sample rate
    start_ts: float      # seconds since this segmenter started running
    end_ts: float
    created_at: float = 0.0


class SileroVAD:
    """Stateful wrapper around Silero VAD v6 ONNX model."""
    
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Default path: backend/audio/silero_vad.onnx
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "silero_vad.onnx")
            
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 4
        
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self.reset()
        
    def reset(self) -> None:
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.context = np.zeros(64, dtype=np.float32)
        self.last_prob = 0.0
        
    def is_speech(self, chunk: np.ndarray, threshold: float = 0.5) -> bool:
        # Silero VAD requires exactly 512 samples for a 32ms chunk at 16kHz
        if len(chunk) < 512:
            chunk = np.pad(chunk, (0, 512 - len(chunk)))
        elif len(chunk) > 512:
            chunk = chunk[:512]
            
        input_data = np.concatenate([self.context, chunk])[np.newaxis, :] # shape (1, 576)
        
        outputs = self.session.run(
            None,
            {"input": input_data, "h": self.h, "c": self.c}
        )
        
        prob = outputs[0][0]
        self.h = outputs[1]
        self.c = outputs[2]
        
        self.context = chunk[-64:]
        self.last_prob = float(prob)
        
        return prob >= threshold



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
        threshold: float = 0.5,
    ):
        self.samplerate = samplerate
        self.frame_ms = frame_ms
        self.frame_samples = int(samplerate * frame_ms / 1000)
        self.vad = SileroVAD()
        self.hangover_frames = max(1, silence_hangover_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.max_segment_frames = int(max_segment_s * 1000 / frame_ms)
        self.rms_threshold = rms_threshold
        self.threshold = threshold
        # ~200-300ms pre-roll (e.g., 288ms if frame_ms=32ms, which is 9 frames)
        self.preroll_frames = max(1, 300 // frame_ms)
        
        import threading
        self.lock = threading.Lock()
        self.active_speech_frames = []
        self.active_speech_start_ts = None
        self.active_speech_silence_run = 0

    def peek_active_segment(self) -> SpeechSegment | None:
        with self.lock:
            actual_speech_len = len(self.active_speech_frames) - self.active_speech_silence_run
            if not self.active_speech_frames or actual_speech_len < self.min_speech_frames:
                return None
            audio = np.concatenate(self.active_speech_frames).astype(np.float32)
            return SpeechSegment(
                audio=audio,
                start_ts=self.active_speech_start_ts if self.active_speech_start_ts is not None else 0.0,
                end_ts=self.active_speech_start_ts if self.active_speech_start_ts is not None else 0.0,
                created_at=time.perf_counter()
            )

    def run(
        self,
        in_queue: "queue.Queue[np.ndarray]",
        out_queue: "queue.Queue[SpeechSegment]",
        stop_event,
        ping_callback=None,
    ) -> None:
        t0 = time.monotonic()
        speech_frames: list[np.ndarray] = []
        speech_probabilities: list[float] = []
        preroll_buffer: list[np.ndarray] = []
        silence_run = 0
        speech_start_ts: float | None = None

        # Reset states at start of stream
        self.vad.reset()

        while not stop_event.is_set():
            if ping_callback:
                try:
                    ping_callback()
                except Exception:
                    pass
            try:
                item = in_queue.get(timeout=0.5)
                if isinstance(item, tuple):
                    timestamp, chunk = item
                else:
                    timestamp, chunk = time.perf_counter(), item
            except queue.Empty:
                continue

            # Ensure correct chunk size
            if len(chunk) < self.frame_samples:
                chunk = np.pad(chunk, (0, self.frame_samples - len(chunk)))
            elif len(chunk) > self.frame_samples:
                chunk = chunk[: self.frame_samples]

            # Silero VAD check
            is_speech = self.vad.is_speech(chunk, self.threshold)
            current_prob = self.vad.last_prob
            
            # Absolute energy gate: if frame energy is below self.rms_threshold, treat as silence.
            if self.rms_threshold > 0.0:
                rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0.0
                if rms < self.rms_threshold:
                    is_speech = False
                    current_prob = 0.0

            now_ts = time.monotonic() - t0

            if is_speech:
                if not speech_frames:
                    # Prepend pre-roll buffer to keep the word onset
                    speech_start_ts = now_ts - len(preroll_buffer) * (self.frame_ms / 1000.0)
                    speech_frames = list(preroll_buffer)
                    speech_probabilities = [0.0] * len(preroll_buffer)
                    preroll_buffer = []
                speech_frames.append(chunk)
                speech_probabilities.append(current_prob)
                silence_run = 0
                
                with self.lock:
                    self.active_speech_frames = list(speech_frames)
                    self.active_speech_start_ts = speech_start_ts
                    self.active_speech_silence_run = silence_run
            elif speech_frames:
                # Keep trailing silence in the buffer for a natural cut-off,
                # but count it toward the hangover threshold.
                speech_frames.append(chunk)
                speech_probabilities.append(current_prob)
                silence_run += 1
                
                with self.lock:
                    self.active_speech_frames = list(speech_frames)
                    self.active_speech_silence_run = silence_run
            else:
                # Rolling buffer of silence during non-speech periods
                preroll_buffer.append(chunk)
                if len(preroll_buffer) > self.preroll_frames:
                    preroll_buffer.pop(0)

            hit_hangover = speech_frames and silence_run >= self.hangover_frames
            # Cut at a natural pause after target length, or unconditionally at 1.5x target length
            hit_max_len = (len(speech_frames) >= self.max_segment_frames and not is_speech) or (len(speech_frames) >= int(self.max_segment_frames * 1.5))
            if speech_frames and (hit_hangover or hit_max_len):
                is_hard_cut = (not hit_hangover) and (len(speech_frames) >= int(self.max_segment_frames * 1.5))
                
                if is_hard_cut:
                    # Look back from the last 20 frames to 5 frames from the end
                    lookback = min(15, len(speech_probabilities) - 5)
                    if lookback > 0:
                        search_start = len(speech_probabilities) - 5 - lookback
                        search_end = len(speech_probabilities) - 5
                        min_rel_idx = int(np.argmin(speech_probabilities[search_start:search_end]))
                        cut_idx = search_start + min_rel_idx
                        
                        audio_frames = speech_frames[:cut_idx + 1]
                        carryover_frames = speech_frames[cut_idx + 1:]
                        carryover_probabilities = speech_probabilities[cut_idx + 1:]
                    else:
                        audio_frames = speech_frames
                        carryover_frames = []
                        carryover_probabilities = []
                else:
                    # Strip trailing silence frames, leaving a comfortable 150ms cushion for word decay
                    cushion_frames = max(1, 150 // self.frame_ms)
                    strip_count = max(0, silence_run - cushion_frames)
                    if strip_count > 0 and len(speech_frames) > strip_count:
                        audio_frames = speech_frames[:-strip_count]
                    else:
                        audio_frames = speech_frames
                    carryover_frames = []
                    carryover_probabilities = []

                if len(audio_frames) - silence_run >= self.min_speech_frames:
                    audio = np.concatenate(audio_frames).astype(np.float32)
                    segment_duration = len(audio_frames) * (self.frame_ms / 1000.0)
                    item = SpeechSegment(
                        audio=audio,
                        start_ts=speech_start_ts,
                        end_ts=speech_start_ts + segment_duration,
                        created_at=time.perf_counter()
                    )
                    try:
                        out_queue.put(item, timeout=2.0)
                    except queue.Full:
                        try:
                            out_queue.get_nowait()
                            out_queue.put_nowait(item)
                        except (queue.Empty, queue.Full):
                            pass
                # Carry over context if the cut was due to maximum duration limit (Soft Commit)
                if hit_max_len:
                    if is_hard_cut:
                        speech_frames = list(carryover_frames)
                        speech_probabilities = list(carryover_probabilities)
                        silence_run = 0
                        speech_start_ts = now_ts - len(speech_frames) * (self.frame_ms / 1000.0)
                        preroll_buffer = []
                    else:
                        carryover_count = max(1, 300 // self.frame_ms)
                        speech_tail = speech_frames[-carryover_count:] if len(speech_frames) >= carryover_count else list(speech_frames)
                        speech_frames = list(speech_tail)
                        speech_probabilities = speech_probabilities[-len(speech_tail):]
                        silence_run = 0
                        speech_start_ts = now_ts - len(speech_tail) * (self.frame_ms / 1000.0)
                        preroll_buffer = []
                    
                    with self.lock:
                        self.active_speech_frames = list(speech_frames)
                        self.active_speech_start_ts = speech_start_ts
                        self.active_speech_silence_run = silence_run
                else:
                    # Seed the preroll buffer for the next utterance using the trailing silence frames
                    # of the current segment, capped at the max preroll size.
                    preroll_buffer = speech_frames[-self.preroll_frames:] if len(speech_frames) >= self.preroll_frames else list(speech_frames)
                    speech_frames = []
                    speech_probabilities = []
                    silence_run = 0
                    speech_start_ts = None
                    
                    with self.lock:
                        self.active_speech_frames = []
                        self.active_speech_start_ts = None
                        self.active_speech_silence_run = 0

        # Flush any remaining active speech frames in the buffer upon shutdown
        if speech_frames:
            actual_speech_len = len(speech_frames) - silence_run
            if actual_speech_len >= self.min_speech_frames:
                audio = np.concatenate(speech_frames).astype(np.float32)
                now_ts = time.monotonic() - t0
                item = SpeechSegment(
                    audio=audio,
                    start_ts=speech_start_ts,
                    end_ts=now_ts,
                    created_at=time.perf_counter()
                )
                try:
                    out_queue.put(item, timeout=2.0)
                except queue.Full:
                    try:
                        out_queue.get_nowait()
                        out_queue.put_nowait(item)
                    except (queue.Empty, queue.Full):
                        pass
