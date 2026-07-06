from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path

# Define directories relative to backend root dynamically
BACKEND_DIR = Path(__file__).resolve().parent

MODELS_DIR = BACKEND_DIR / "models"
HISTORY_DIR = BACKEND_DIR / "history"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# Set Hugging Face environment variables to redirect model downloads directly to backend/models directory
os.environ["HF_HOME"] = str(MODELS_DIR)
os.environ["HF_HUB_CACHE"] = str(MODELS_DIR)
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
    
    # Model download folder inside backend
    model_download_root: str = str(MODELS_DIR)

    # --- Output ---
    transcript_log_path: str = str(BACKEND_DIR / "transcript.log")
    history_dir: str = str(HISTORY_DIR)

