from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path

# Direct all downloads and storage to the E drive
PROJECT_ROOT = Path("E:/Local transcribe/local-transcribe")

# Ensure directories exist
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / ".cache").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "history").mkdir(parents=True, exist_ok=True)

# Set Hugging Face environment variables to redirect model downloads to the E drive
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
os.environ["HF_HUB_CACHE"] = str(PROJECT_ROOT / ".cache" / "huggingface" / "hub")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Suppress annoying warning messages from third-party libraries to clean up console output
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*cache-system uses symlinks.*")
warnings.filterwarnings("ignore", message=".*data discontinuity in recording.*")



@dataclass
class Config:
    # --- Audio ---
    sample_rate: int = 16000           # required by Whisper and webrtcvad
    frame_ms: int = 30                 # VAD frame size, webrtcvad only accepts 10/20/30ms
    mic_device: Optional[str] = None   # None = OS default microphone
    speaker_device: Optional[str] = None  # None = OS default output device (loopback source)

    # --- VAD / endpointing ---
    vad_aggressiveness: int = 2        # 0-3, higher = more aggressive at filtering non-speech
    silence_hangover_ms: int = 500     # trailing silence required before a segment is closed
    min_speech_ms: int = 250           # ignore blips shorter than this (coughs, clicks)
    max_segment_s: float = 10.0        # hard limit for segment duration, cut only at silent boundaries

    # --- STT ---
    engine_type: str = "moonshine"     # "faster-whisper" | "moonshine"
    model_size: str = "moonshine/base"  # try "moonshine/tiny" if this lags or for lower memory
    compute_type: str = "float"         # "float" for Moonshine, "int8" for Whisper
    device: str = "dml"                # "dml" | "cpu" | "cuda"
    
    # Model download folder on the E drive
    model_download_root: str = str(PROJECT_ROOT / "models")

    # --- Output ---
    transcript_log_path: str = str(PROJECT_ROOT / "transcript.log")
    history_dir: str = str(PROJECT_ROOT / "history")

