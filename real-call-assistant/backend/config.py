from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path

# Define directories relative to backend root dynamically
BACKEND_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file if it exists
_env_path = BACKEND_DIR / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ[_key.strip()] = _val.strip()


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

    # --- AEC (Acoustic Echo Cancellation) ---
    enable_aec: bool = True
    aec_delay_ms: int = 80             # acoustic render-to-capture latency in ms
    aec_enable_ns: bool = False
    aec_enable_agc: bool = False
    aec_ducking_threshold: float = 0.45 # mic_rms / ref_rms ratio below which we duck
    aec_ref_threshold: float = 0.01     # speaker RMS above which we consider speaker active


    # --- VAD / endpointing ---
    vad_aggressiveness: int = 2        # 0-3, higher = more aggressive at filtering non-speech
    silence_hangover_ms: int = 500     # trailing silence required before a segment is closed
    min_speech_ms: int = 250           # ignore blips shorter than this (coughs, clicks)
    max_segment_s: float = 4.0         # hard limit for segment duration, cut only at silent boundaries
    vad_rms_threshold: float = 0.008   # absolute RMS threshold below which frames are treated as silence

    # --- STT ---
    # URL of the remote STT server (RTX 5090 machine via Tailscale)
    remote_url: str = os.getenv("STT_REMOTE_URL", "")
    
    engine_type: str = os.getenv("STT_ENGINE_TYPE", "moonshine")  # "faster-whisper" | "moonshine" | "remote"
    model_size: str = os.getenv("STT_MODEL_SIZE", "moonshine/base") # e.g. "remote/distil-large-v3"
    compute_type: str = "float"         # "float" for Moonshine, "int8" for Whisper, "remote" for Remote
    device: str = "dml"                # "dml" | "cpu" | "cuda" | "remote"
    stt_language: str = os.getenv("STT_LANGUAGE", "en")
    stt_initial_prompt: str = os.getenv("STT_INITIAL_PROMPT", "Indian English accent, conversation, terminology.")
    
    # Model download folder inside backend
    model_download_root: str = str(MODELS_DIR)

    # --- Output ---
    transcript_log_path: str = str(BACKEND_DIR / "transcript.log")
    history_dir: str = str(HISTORY_DIR)


