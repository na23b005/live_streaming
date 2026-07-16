"""
faster-whisper (CTranslate2) backed STT engine.

Honest hardware note: CTranslate2, the inference engine faster-whisper wraps,
only accelerates on NVIDIA CUDA or CPU. It has no AMD ROCm or DirectML
backend, so on an AMD card (like your RX 6650) this will resolve to "cpu"
below - which is fine, since tiny/base/small models run comfortably
real-time on a modern CPU, matching your "try smaller models first" plan.

When you're ready for real AMD (and cross-vendor) GPU acceleration, swap this
module for one built on whisper.cpp compiled with its Vulkan backend
(GGML_VULKAN=1) - Vulkan compute works on AMD, NVIDIA, and Intel GPUs on both
Windows and Linux. Because everything implements STTEngine, that's a
new file, not a rewrite. See README "GPU acceleration path" section.
"""

import numpy as np
from faster_whisper import WhisperModel

from .base import STTEngine


def resolve_device(requested: str) -> tuple[str, str]:
    """Returns (device, default_compute_type)."""
    if requested == "cpu":
        return "cpu", "int8"
    if requested == "cuda":
        return "cuda", "float16"

    # auto: use CUDA only if it's actually available (NVIDIA only).
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


class FasterWhisperEngine(STTEngine):
    def __init__(self, model_size: str = "base.en", device: str = "auto", compute_type: str | None = None, download_root: str | None = None, language: str | None = None, initial_prompt: str | None = None, hotwords: str | None = None, prefix: str | None = None):
        resolved_device, default_compute = resolve_device(device)
        self.model = WhisperModel(
            model_size,
            device=resolved_device,
            compute_type=compute_type or default_compute,
            download_root=download_root,
        )
        self.device = resolved_device
        self.language = language
        self.initial_prompt = initial_prompt
        self.hotwords = hotwords
        self.prefix = prefix
        self.total_transcribe_time = 0.0
        self.total_segments = 0
        self.total_audio_duration = 0.0

        # Warm up model to compile kernels and warm caches (pre-empt first-utterance latency)
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        _ = self.transcribe(dummy_audio, 16000)

    def transcribe(self, audio: np.ndarray, samplerate: int, meeting_id: str | None = None) -> str:
        if samplerate != 16000:
            raise ValueError("faster-whisper expects 16kHz audio; resample before calling transcribe().")
        
        # Root mean square (RMS) threshold to filter out silence/ambient hum
        rms = np.sqrt(np.mean(audio**2)) if len(audio) > 0 else 0.0
        if rms < 0.006:
            return ""
            
        import time
        start_t = time.perf_counter()
        
        lang = self.language if self.language else None
        prompt = self.initial_prompt if self.initial_prompt else None
        prefix = self.prefix if self.prefix else None
        hotwords = self.hotwords if self.hotwords else None
        
        segments, _info = self.model.transcribe(
            audio,
            beam_size=5,
            language=lang,
            initial_prompt=prompt,
            prefix=prefix,
            hotwords=hotwords,
            vad_filter=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.85,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.5,
            temperature=0.0,
        )
        
        texts = []
        for seg in segments:
            # Layer 2b: Whisper's own no-speech detection (relaxed for Indian accent/code-switching)
            if seg.no_speech_prob > 0.85:
                continue  # Skip: Whisper thinks this is silence/noise
            
            # Layer 2c: Repetition detection via compression ratio (stuttering)
            if seg.compression_ratio > 2.4:
                continue  # Skip: likely stuck in repetition loop
            
            # Layer 2d: Low-confidence detection (relaxed floor for Indian accent)
            if seg.avg_logprob < -1.5:
                continue  # Skip: Whisper is guessing
                
            # Filter out common silence hallucinations if confidence is low
            cleaned_text = seg.text.strip().lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "")
            if cleaned_text in ("thank you", "thank you for watching"):
                if seg.avg_logprob < -1.0 or seg.no_speech_prob > 0.6:
                    continue  # Skip: likely hallucination on silence/noise
                
            # Store raw audio segments for segments with avg_logprob < -1.0 so the user can re-transcribe them later
            if meeting_id and seg.avg_logprob < -1.0:
                import scipy.io.wavfile as wav
                import os
                from config import HISTORY_DIR
                
                start_idx = int(seg.start * samplerate)
                end_idx = int(seg.end * samplerate)
                seg_audio = audio[start_idx:end_idx]
                if len(seg_audio) > 0:
                    debug_dir = os.path.join(HISTORY_DIR, f"{meeting_id}_low_conf")
                    os.makedirs(debug_dir, exist_ok=True)
                    filename = f"seg_{seg.start:.2f}_{seg.end:.2f}_prob_{seg.avg_logprob:.2f}.wav"
                    wav_path = os.path.join(debug_dir, filename)
                    try:
                        wav.write(wav_path, samplerate, (np.clip(seg_audio, -1.0, 1.0) * 32767).astype(np.int16))
                        print(f"[LowConf] Saved low-confidence segment to {wav_path}")
                    except Exception as ex:
                        print(f"[LowConf] Failed to save WAV segment: {ex}")

            texts.append(seg.text.strip())
        text = " ".join(texts).strip()
        
        duration = time.perf_counter() - start_t
        self.total_transcribe_time += duration
        self.total_segments += 1
        self.total_audio_duration += len(audio) / samplerate
        
        return text
