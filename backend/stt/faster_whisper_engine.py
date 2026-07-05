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
    def __init__(self, model_size: str = "base.en", device: str = "auto", compute_type: str | None = None, download_root: str | None = None):
        resolved_device, default_compute = resolve_device(device)
        self.model = WhisperModel(
            model_size,
            device=resolved_device,
            compute_type=compute_type or default_compute,
            download_root=download_root,
        )
        self.device = resolved_device
        self.total_transcribe_time = 0.0
        self.total_segments = 0
        self.total_audio_duration = 0.0

    def transcribe(self, audio: np.ndarray, samplerate: int) -> str:
        if samplerate != 16000:
            raise ValueError("faster-whisper expects 16kHz audio; resample before calling transcribe().")
        
        import time
        start_t = time.perf_counter()
        
        segments, _info = self.model.transcribe(audio, beam_size=1, vad_filter=False)
        
        texts = []
        for seg in segments:
            # Skip segments with high probability of no speech or low confidence (hallucinations/typing clicks)
            if seg.no_speech_prob > 0.65 or seg.avg_logprob < -1.0:
                continue
            texts.append(seg.text.strip())
        text = " ".join(texts).strip()
        
        duration = time.perf_counter() - start_t
        self.total_transcribe_time += duration
        self.total_segments += 1
        self.total_audio_duration += len(audio) / samplerate
        
        return text
