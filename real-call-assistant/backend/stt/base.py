"""
Abstract STT engine interface.

Everything downstream (ChannelWorker) only talks to this interface, not to
faster-whisper directly. That's the seam where you'd later plug in a real
AMD-GPU-accelerated backend (e.g. whisper.cpp built with the Vulkan
backend, which - unlike faster-whisper/CTranslate2 - actually runs on
AMD/NVIDIA/Intel GPUs) without touching audio/VAD/pipeline code at all.
"""

from abc import ABC, abstractmethod

import numpy as np


class STTEngine(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, samplerate: int, meeting_id: str | None = None) -> str:
        """Return transcribed text for a mono float32 audio array."""
        raise NotImplementedError
