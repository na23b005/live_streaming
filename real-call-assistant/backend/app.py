import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*discontinuity in recording.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import asyncio
import queue
import threading
import uvicorn
import json
import datetime
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import config first so Hugging Face environment variables are set before other modules load.
from config import Config
from audio.capture import MicCapture, SystemAudioCapture
from audio.vad_segmenter import VADSegmenter, SileroVAD
from audio.echo_cancellation import WebRTCAcousticEchoCanceller, calibrate_aec_delay
from pipeline.channel_worker import ChannelWorker, TranscriptEvent
from pipeline.transcript_normalizer import format_line
from stt.moonshine_dml_engine import MoonshineDirectMLEngine
from stt.faster_whisper_engine import FasterWhisperEngine
from stt.remote_engine import RemoteSTTEngine
from stt.nemotron_remote_engine import NemotronRemoteEngine
import webview
import subprocess

app = FastAPI()

# Enable CORS for React frontend (development and Tauri)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
cfg = Config()
mic_engine = None
sys_engine = None
mic_worker = None
sys_worker = None
out_queue = queue.Queue()
recording_active = False
active_sockets = []
current_meeting_id = None

is_loading_engine = False
engine_load_error = None

# Global memory storage for the active transcription session
current_session_segments = []
current_session_start_time = None

class RenameRequest(BaseModel):
    title: str

def get_system_specs():
    specs = {
        "gpu": [],
        "has_dml": False,
        "has_cuda": False,
        "ram_gb": 8.0,
    }
    
    # Check ONNX Runtime providers
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        specs["has_dml"] = "DmlExecutionProvider" in providers
        specs["has_cuda"] = "CUDAExecutionProvider" in providers
    except Exception:
        pass
        
    # Get GPU names via WMIC
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        out = subprocess.check_output(
            ["wmic", "path", "win32_VideoController", "get", "name"],
            startupinfo=startupinfo,
            text=True
        )
        gpus = []
        for line in out.splitlines():
            line = line.strip()
            if line and not line.lower().startswith("name"):
                if line not in gpus:
                    gpus.append(line)
        specs["gpu"] = gpus
    except Exception:
        pass
        
    # Get RAM via WMIC
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        out = subprocess.check_output(
            ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
            startupinfo=startupinfo,
            text=True
        )
        lines = []
        for line in out.splitlines():
            line = line.strip()
            if line and not line.lower().startswith("total"):
                lines.append(line)
        if lines:
            mem_bytes = int(lines[0])
            specs["ram_gb"] = round(mem_bytes / (1024 ** 3), 1)
    except Exception:
        pass
        
    return specs

def get_model_recommendation():
    specs = get_system_specs()
    gpu_names = specs["gpu"]
    ram_gb = specs["ram_gb"]
    
    # Check if we have dedicated GPU or good acceleration
    has_dedicated_gpu = False
    gpu_desc = "None"
    
    if gpu_names:
        gpu_desc = ", ".join(gpu_names)
        # Check if the name indicates a discrete or high-power GPU
        for name in gpu_names:
            name_lower = name.lower()
            if any(term in name_lower for term in ["nvidia", "geforce", "rtx", "gtx", "amd", "radeon", "intel arc"]):
                # Filter out standard basic display adapter
                if "microsoft basic display adapter" not in name_lower:
                    has_dedicated_gpu = True
                    
    # Recommendation logic maps to one of our listed model IDs
    if (specs["has_dml"] or specs["has_cuda"] or has_dedicated_gpu) and ram_gb >= 12.0:
        recommended = "moonshine/base"
        reason = f"Detected GPU ({gpu_desc}) and {ram_gb} GB of RAM. Moonshine Base is recommended for high accuracy and fast latency."
    else:
        recommended = "moonshine/tiny"
        reason = f"Detected {ram_gb} GB of RAM. Moonshine Tiny is recommended for fast English transcription with low CPU/memory usage."
        
    return {
        "gpu_info": gpu_names,
        "ram_gb": ram_gb,
        "has_gpu_accel": specs["has_dml"] or specs["has_cuda"] or has_dedicated_gpu,
        "recommended_model": recommended,
        "recommended_reason": reason
    }

MODELS_METADATA = [
    {
        "id": "moonshine/tiny",
        "name": "Moonshine Tiny",
        "size": "26 MB",
        "speed": "very-fast",
        "accuracy": "good acc"
    },
    {
        "id": "moonshine/base",
        "name": "Moonshine Base",
        "size": "60 MB",
        "speed": "very-fast",
        "accuracy": "very-high acc"
    },
    {
        "id": "tiny.en",
        "name": "Whisper Tiny EN",
        "size": "75 MB",
        "speed": "very-fast",
        "accuracy": "good acc"
    },
    {
        "id": "base.en",
        "name": "Whisper Base EN",
        "size": "142 MB",
        "speed": "fast",
        "accuracy": "very-high acc"
    },
    {
        "id": "distil-medium.en",
        "name": "Distil Medium EN",
        "size": "789 MB",
        "speed": "fast",
        "accuracy": "very-high acc"
    },
    {
        "id": "distil-large-v3",
        "name": "Distil Large v3",
        "size": "1.51 GB",
        "speed": "medium",
        "accuracy": "very-high acc"
    },
    {
        "id": "large-v3-turbo",
        "name": "Whisper Large v3 Turbo",
        "size": "1.6 GB",
        "speed": "medium",
        "accuracy": "excellent acc"
    },
    {
        "id": "Tejveer12/Indian-Accent-English-Whisper-Finetuned",
        "name": "Indian English (Accent-Finetuned)",
        "size": "1.6 GB",
        "speed": "medium",
        "accuracy": "excellent acc"
    },
    {
        "id": "Trelis/whisper-hinglish-preview",
        "name": "Hinglish (Code-Switched)",
        "size": "3.1 GB",
        "speed": "slow",
        "accuracy": "excellent acc"
    },
    {
        "id": "ai4bharat/whisper-medium-en-indic",
        "name": "Indic-English (Medium size)",
        "size": "1.5 GB",
        "speed": "medium",
        "accuracy": "very-high acc"
    },
    {
        "id": "remote/distil-large-v3",
        "name": "Remote GPU (RTX 5090) - Distil Large v3",
        "size": "1.51 GB",
        "speed": "blazing-fast",
        "accuracy": "excellent acc"
    },
    {
        "id": "remote/large-v3-turbo",
        "name": "Remote GPU (RTX 5090) - Whisper Large v3 Turbo",
        "size": "1.6 GB",
        "speed": "blazing-fast",
        "accuracy": "excellent acc"
    },
    {
        "id": "remote/Tejveer12/Indian-Accent-English-Whisper-Finetuned",
        "name": "Remote GPU (RTX 5090) - Indian English",
        "size": "1.6 GB",
        "speed": "blazing-fast",
        "accuracy": "excellent acc"
    },
    {
        "id": "remote/Trelis/whisper-hinglish-preview",
        "name": "Remote GPU (RTX 5090) - Hinglish",
        "size": "3.1 GB",
        "speed": "blazing-fast",
        "accuracy": "excellent acc"
    },
    {
        "id": "remote/ai4bharat/whisper-medium-en-indic",
        "name": "Remote GPU (RTX 5090) - Indic-English",
        "size": "1.5 GB",
        "speed": "blazing-fast",
        "accuracy": "very-high acc"
    },
    {
        "id": "remote/nvidia-nemotron-3.5",
        "name": "Remote GPU (RTX 5090) - Nemotron 3.5 ASR",
        "size": "1.2 GB",
        "speed": "blazing-fast",
        "accuracy": "excellent acc"
    }
]


from pathlib import Path
import shutil
from tqdm.auto import tqdm

downloading_models = set()
cancelled_downloads = set()
download_progress = {}
download_bytes_tracker = {}

class HubDownloadProgress(tqdm):
    def __init__(self, *args, model_id=None, total_size=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = model_id
        self.total_size = total_size
        
    def update(self, n=1):
        global cancelled_downloads
        if self.model_id and self.model_id in cancelled_downloads:
            raise RuntimeError("Download cancelled by user")
            
        super().update(n)
        global download_bytes_tracker, download_progress
        if self.model_id:
            download_bytes_tracker[self.model_id] = download_bytes_tracker.get(self.model_id, 0) + n
            if self.total_size > 0:
                percent = int((download_bytes_tracker[self.model_id] / self.total_size) * 100)
                download_progress[self.model_id] = max(0, min(100, percent))

MODEL_REPOS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base.en": "Systran/faster-whisper-base.en",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "large-v3-turbo": "Systran/faster-whisper-large-v3-turbo",
    "moonshine/tiny": "UsefulSensors/moonshine",
    "moonshine/base": "UsefulSensors/moonshine",
}

def check_model_downloaded(model_id: str) -> bool:
    if model_id.startswith("remote/"):
        return True
    hf_cache_dir = Path(os.environ.get("HF_HUB_CACHE", "E:/Local transcribe/local-transcribe/models"))
    if model_id == "moonshine/tiny":
        moonshine_repo = hf_cache_dir / "models--UsefulSensors--moonshine"
        if moonshine_repo.exists():
            for p in moonshine_repo.glob("**/tiny/**/encoder_model.onnx"):
                return True
        return False
    elif model_id == "moonshine/base":
        moonshine_repo = hf_cache_dir / "models--UsefulSensors--moonshine"
        if moonshine_repo.exists():
            for p in moonshine_repo.glob("**/base/**/encoder_model.onnx"):
                return True
        return False
    elif "/" in model_id:
        safe_name = model_id.replace("/", "--")
        models_dir = Path(cfg.model_download_root)
        out_dir = models_dir / f"models--{safe_name}-ct2"
        if out_dir.exists():
            for p in out_dir.glob("**/model.bin"):
                return True
        return False
    else:
        models_dir = Path(cfg.model_download_root)
        repo_id = MODEL_REPOS.get(model_id, model_id)
        safe_name = repo_id.replace("/", "--")
        repo_dir = models_dir / f"models--{safe_name}"
        if repo_dir.exists():
            for p in repo_dir.glob("**/model.bin"):
                return True
        return False

def download_model_worker(model_id: str):
    global downloading_models, download_progress, download_bytes_tracker
    try:
        print(f"Starting download for model '{model_id}'...")
        download_progress[model_id] = 0
        download_bytes_tracker[model_id] = 0
        
        repo_id = MODEL_REPOS.get(model_id, model_id)
        
        # Get total size programmatically using HfApi with a retry loop for transient timeouts (502/504)
        from huggingface_hub import HfApi
        import time
        total_size = 0
        for attempt in range(3):
            try:
                api = HfApi()
                info = api.model_info(repo_id, files_metadata=True)
                total_size = sum(f.size for f in info.siblings if f.size)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"Warning: could not get repo info for progress calculation after 3 attempts: {e}")
                else:
                    time.sleep(1.5)
            
        # Dynamically subclass HubDownloadProgress to avoid 'functools.partial' object has no attribute 'get_lock' error in tqdm thread_map
        class DynamicHubDownloadProgress(HubDownloadProgress):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, model_id=model_id, total_size=total_size, **kwargs)
        
        progress_class = DynamicHubDownloadProgress
        
        from huggingface_hub import snapshot_download
        models_dir = Path(cfg.model_download_root)
        
        if model_id.startswith("moonshine/"):
            # Moonshine: download using snapshot_download with the custom progress
            snapshot_download(
                repo_id=repo_id,
                cache_dir=str(models_dir),
                tqdm_class=progress_class
            )
        elif "/" in model_id:
            # Custom model: e.g. Tejveer12/Indian-Accent-English-Whisper-Finetuned
            safe_name = model_id.replace("/", "--")
            raw_dir = models_dir / f"models--{safe_name}-raw"
            out_dir = models_dir / f"models--{safe_name}-ct2"
            
            snapshot_download(
                repo_id=repo_id,
                local_dir=raw_dir,
                tqdm_class=progress_class
            )
            
            # Show 99% progress when starting CTranslate2 conversion
            download_progress[model_id] = 99
            
            import sys
            import subprocess
            print(f"Converting PyTorch model '{model_id}' to CTranslate2...")
            subprocess.run([
                sys.executable, "-m", "ctranslate2.converters.transformers",
                "--model", str(raw_dir),
                "--output_dir", str(out_dir),
                "--copy_files", "tokenizer.json", "preprocessor_config.json",
                "--quantization", "int8"
            ], check=True)
            
            shutil.rmtree(raw_dir)
        else:
            # Standard Systran model: download directly via snapshot_download
            snapshot_download(
                repo_id=repo_id,
                cache_dir=str(models_dir),
                tqdm_class=progress_class
            )
            
        download_progress[model_id] = 100
        print(f"Finished downloading model '{model_id}' successfully.")
    except Exception as e:
        print(f"Error downloading model '{model_id}': {e}")
        global cancelled_downloads
        if model_id in cancelled_downloads:
            print(f"Cleaning up files for cancelled model '{model_id}'...")
            try:
                delete_model_files(model_id)
            except Exception as cleanup_err:
                print(f"Error cleaning up files for cancelled model: {cleanup_err}")
            cancelled_downloads.remove(model_id)
        # Clear progress on failure
        if model_id in download_progress:
            del download_progress[model_id]
    finally:
        if model_id in downloading_models:
            downloading_models.remove(model_id)

def delete_model_files(model_id: str):
    if model_id.startswith("remote/"):
        return
    models_dir = Path(cfg.model_download_root)
    hf_cache_dir = Path(os.environ.get("HF_HUB_CACHE", str(models_dir)))
    
    # 1. Clean up locks if any
    try:
        locks_dir = hf_cache_dir / ".locks"
        if locks_dir.exists():
            for p in list(locks_dir.glob("**/*")):
                if p.is_file() and model_id in p.name:
                    p.unlink(missing_ok=True)
    except Exception as e:
        print(f"Warning: could not clear locks for '{model_id}': {e}")
        
    # 2. Delete the actual directories
    if model_id.startswith("moonshine/"):
        for name in ["models--useful-sensors--moonshine", "models--UsefulSensors--moonshine"]:
            repo_dir = hf_cache_dir / name
            if repo_dir.exists():
                shutil.rmtree(repo_dir, ignore_errors=True)
    elif "/" in model_id and not model_id.startswith("moonshine/"):
        # Custom converted models
        safe_name = model_id.replace("/", "--")
        raw_dir = models_dir / f"models--{safe_name}-raw"
        out_dir = models_dir / f"models--{safe_name}-ct2"
        if raw_dir.exists():
            shutil.rmtree(raw_dir, ignore_errors=True)
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
    else:
        # Standard Systran models
        repo_id = MODEL_REPOS.get(model_id, model_id)
        safe_name = repo_id.replace("/", "--")
        repo_dir = models_dir / f"models--{safe_name}"
        if repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)

def load_engines(model_size: str):
    global mic_engine, sys_engine, cfg
    
    # Determine the engine type and parameters
    if model_size.startswith("remote/"):
        engine_type = "remote"
        device = "remote"
        compute_type = "remote"
    elif model_size.startswith("moonshine/"):
        engine_type = "moonshine"
        device = "dml"  # default DirectML for Moonshine on Windows
        compute_type = "float"
    else:
        engine_type = "faster-whisper"
        device = "auto"  # CUDA if available, else CPU
        compute_type = "int8"  # Whisper on CPU or auto
        
    print(f"Loading {engine_type} STT engines for model '{model_size}' (device={device})...")
    
    if engine_type == "remote":
        # Extract the backend model size (e.g. distil-large-v3)
        actual_model = model_size.replace("remote/", "")
        if "nemotron" in actual_model:
            new_mic_engine = NemotronRemoteEngine(
                model_size=actual_model,
                remote_url=cfg.remote_url,
                language=cfg.stt_language
            )
            new_sys_engine = NemotronRemoteEngine(
                model_size=actual_model,
                remote_url=cfg.remote_url,
                language=cfg.stt_language
            )
        else:
            new_mic_engine = RemoteSTTEngine(
                model_size=actual_model,
                remote_url=cfg.remote_url,
                language=cfg.stt_language,
                initial_prompt=cfg.stt_initial_prompt,
                hotwords=cfg.stt_hotwords,
                prefix=cfg.stt_prefix
            )
            new_sys_engine = RemoteSTTEngine(
                model_size=actual_model,
                remote_url=cfg.remote_url,
                language=cfg.stt_language,
                initial_prompt=cfg.stt_initial_prompt,
                hotwords=cfg.stt_hotwords,
                prefix=cfg.stt_prefix
            )
    elif engine_type == "moonshine":
        new_mic_engine = MoonshineDirectMLEngine(
            model_size,
            device,
            compute_type=compute_type,
            download_root=cfg.model_download_root,
        )
        new_sys_engine = MoonshineDirectMLEngine(
            model_size,
            device,
            compute_type=compute_type,
            download_root=cfg.model_download_root,
        )
    else:
        # Check if it is a local custom model
        load_path = model_size
        if "/" in model_size and not model_size.startswith("moonshine/"):
            safe_name = model_size.replace("/", "--")
            models_dir = Path(cfg.model_download_root)
            out_dir = models_dir / f"models--{safe_name}-ct2"
            if out_dir.exists():
                load_path = str(out_dir)
                
        new_mic_engine = FasterWhisperEngine(
            load_path,
            device,
            compute_type,
            download_root=cfg.model_download_root,
            language=cfg.stt_language,
            initial_prompt=cfg.stt_initial_prompt,
            hotwords=cfg.stt_hotwords,
            prefix=cfg.stt_prefix
        )
        new_sys_engine = FasterWhisperEngine(
            load_path,
            device,
            compute_type,
            download_root=cfg.model_download_root,
            language=cfg.stt_language,
            initial_prompt=cfg.stt_initial_prompt,
            hotwords=cfg.stt_hotwords,
            prefix=cfg.stt_prefix
        )
        
    # Update config
    cfg.engine_type = engine_type
    cfg.model_size = model_size
    cfg.device = device
    cfg.compute_type = compute_type
    
    # Swap engines
    mic_engine = new_mic_engine
    sys_engine = new_sys_engine
    print(f"STT engines loaded successfully. Active Device: {mic_engine.device}")

def bg_load_engines(model_size: str):
    global is_loading_engine, engine_load_error
    is_loading_engine = True
    engine_load_error = None
    try:
        load_engines(model_size)
    except Exception as e:
        engine_load_error = str(e)
        print(f"Failed to load engines for model '{model_size}': {e}")
    finally:
        is_loading_engine = False

def build_segmenter(cfg: Config) -> VADSegmenter:
    return VADSegmenter(
        samplerate=cfg.sample_rate,
        frame_ms=cfg.frame_ms,
        aggressiveness=cfg.vad_aggressiveness,
        silence_hangover_ms=cfg.silence_hangover_ms,
        min_speech_ms=cfg.min_speech_ms,
        max_segment_s=cfg.max_segment_s,
        rms_threshold=cfg.vad_rms_threshold,
        threshold=cfg.vad_threshold,
    )

def auto_save_loop():
    global recording_active, current_session_segments, current_session_start_time, current_meeting_id, cfg
    import time
    import datetime
    import json
    import os
    while True:
        time.sleep(30)
        if recording_active and current_meeting_id:
            try:
                duration = 0.0
                if current_session_start_time:
                    duration = (datetime.datetime.now() - current_session_start_time).total_seconds()
                
                meeting_data = {
                    "id": current_meeting_id,
                    "title": "Active Meeting",
                    "date": current_session_start_time.isoformat() if current_session_start_time else datetime.datetime.now().isoformat(),
                    "duration": duration,
                    "segments": list(current_session_segments),
                    "status": "active"
                }
                os.makedirs(cfg.history_dir, exist_ok=True)
                file_path = os.path.join(cfg.history_dir, "active_meeting.json")
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(meeting_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[AutoSave] Error saving active meeting: {e}")

@app.on_event("startup")
def startup_event():
    global cfg
    print("Initializing STT engines in background...")
    import threading
    threading.Thread(target=bg_load_engines, args=(cfg.model_size,), daemon=True).start()
    threading.Thread(target=auto_save_loop, daemon=True).start()

@app.post("/api/start")
def start_recording():
    global mic_worker, sys_worker, recording_active, out_queue
    global current_session_segments, current_session_start_time, current_meeting_id
    if recording_active:
        return {"status": "already_recording"}
        
    # Clear the queue
    while not out_queue.empty():
        try:
            out_queue.get_nowait()
        except queue.Empty:
            break
            
    # Reset active session store
    current_session_segments = []
    current_session_start_time = datetime.datetime.now()
    meeting_id = f"meeting_{current_session_start_time.strftime('%Y%m%d_%H%M%S')}"
    current_meeting_id = meeting_id

    # Write initial active_meeting.json to disk
    try:
        meeting_data = {
            "id": meeting_id,
            "title": "Active Meeting",
            "date": current_session_start_time.isoformat(),
            "duration": 0.0,
            "segments": [],
            "status": "active"
        }
        os.makedirs(cfg.history_dir, exist_ok=True)
        file_path = os.path.join(cfg.history_dir, "active_meeting.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(meeting_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save initial active meeting state: {e}")
            
    # Initialize echo canceller if enabled
    echo_canceller = None
    if cfg.enable_aec:
        measured_delay = calibrate_aec_delay(
            sample_rate=cfg.sample_rate,
            mic_device_name=cfg.mic_device,
            speaker_device_name=cfg.speaker_device,
        )
        echo_canceller = WebRTCAcousticEchoCanceller(
            sample_rate=cfg.sample_rate,
            delay_ms=measured_delay,
            enable_ns=cfg.aec_enable_ns,
            enable_agc=cfg.aec_enable_agc,
            ducking_threshold=cfg.aec_ducking_threshold,
            ref_threshold=cfg.aec_ref_threshold,
            correlation_threshold=cfg.aec_correlation_threshold,
            correlation_lag_search_ms=cfg.aec_correlation_lag_search_ms,
            correlation_window_ms=cfg.aec_correlation_window_ms,
        )

    shared_state = {"last_sys_audio_active_time": 0.0}

    # Re-initialize workers
    mic_worker = ChannelWorker(
        speaker_label="Speaker 1",
        capture=MicCapture(cfg.sample_rate, cfg.frame_ms, cfg.mic_device),
        segmenter=build_segmenter(cfg),
        engine=mic_engine,
        out_queue=out_queue,
        echo_canceller=echo_canceller,
        shared_state=shared_state,
        meeting_id=meeting_id
    )
    sys_worker = ChannelWorker(
        speaker_label="Speaker 2",
        capture=SystemAudioCapture(cfg.sample_rate, cfg.frame_ms, cfg.speaker_device),
        segmenter=build_segmenter(cfg),
        engine=sys_engine,
        out_queue=out_queue,
        echo_canceller=echo_canceller,
        shared_state=shared_state,
        meeting_id=meeting_id
    )
    
    # Reset stats
    mic_engine.total_segments = 0
    mic_engine.total_audio_duration = 0.0
    mic_engine.total_transcribe_time = 0.0
    
    sys_engine.total_segments = 0
    sys_engine.total_audio_duration = 0.0
    sys_engine.total_transcribe_time = 0.0
    
    # Start workers
    mic_worker.start()
    sys_worker.start()
    recording_active = True
    print("Recording started.")
    return {"status": "started"}

def reprocess_recording(mic_audio, sys_audio, cfg, mic_engine, sys_engine):
    from audio.vad_segmenter import SileroVAD
    from pipeline.channel_worker import stt_lock
    from pipeline.transcript_normalizer import clean_text, is_whisper_hallucination
    
    # 1. Pre-calculate system loopback activity timestamps for room echo masking
    sys_active_times = []
    frame_samples = int(cfg.sample_rate * cfg.frame_ms / 1000)
    t = 0.0
    for idx in range(0, len(sys_audio), frame_samples):
        chunk = sys_audio[idx : idx + frame_samples]
        rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0.0
        if rms > 0.01:
            sys_active_times.append(t)
        t += cfg.frame_ms / 1000.0
        
    def get_speech_segments_sync(audio_array, is_mic=False):
        if len(audio_array) == 0:
            return []
            
        vad = SileroVAD()
        # Use a larger silence hangover (800ms) for post-processing to group continuous speech
        post_hangover_ms = 800
        hangover_frames = max(1, post_hangover_ms // cfg.frame_ms)
        min_speech_frames = max(1, cfg.min_speech_ms // cfg.frame_ms)
        # Use a larger max segment limit (25.0s) to keep full sentence structures together
        post_max_segment_s = 25.0
        max_segment_frames = int(post_max_segment_s * 1000 / cfg.frame_ms)
        
        speech_frames = []
        speech_probabilities = []
        preroll_frames = max(1, 300 // cfg.frame_ms)
        preroll_buffer = []
        silence_run = 0
        speech_start_ts = None
        
        segments = []
        t_offset = 0.0
        
        for idx in range(0, len(audio_array), frame_samples):
            chunk = audio_array[idx : idx + frame_samples]
            if len(chunk) < frame_samples:
                chunk = np.pad(chunk, (0, frame_samples - len(chunk)))
                
            rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0.0
            
            is_speech = vad.is_speech(chunk, cfg.vad_threshold)
            current_prob = vad.last_prob
            
            # Dynamic threshold masking for microphone channel to suppress room echo leakage
            current_rms_threshold = 0.012 if is_mic else 0.008
            if is_mic and sys_active_times:
                # Check if system loopback was active in the last 1.0s
                was_sys_active = any(t_offset - 1.0 <= st <= t_offset for st in sys_active_times)
                if was_sys_active:
                    current_rms_threshold = 0.035
                    
            if rms < current_rms_threshold:
                is_speech = False
                current_prob = 0.0
                
            if is_speech:
                if not speech_frames:
                    # Prepend pre-roll buffer to keep the word onset
                    speech_start_ts = t_offset - len(preroll_buffer) * (cfg.frame_ms / 1000.0)
                    speech_frames = list(preroll_buffer)
                    speech_probabilities = [0.0] * len(preroll_buffer)
                    preroll_buffer = []
                speech_frames.append(chunk)
                speech_probabilities.append(current_prob)
                silence_run = 0
            elif speech_frames:
                speech_frames.append(chunk)
                speech_probabilities.append(current_prob)
                silence_run += 1
            else:
                preroll_buffer.append(chunk)
                if len(preroll_buffer) > preroll_frames:
                    preroll_buffer.pop(0)
                
            hit_hangover = speech_frames and silence_run >= hangover_frames
            hit_max_len = (len(speech_frames) >= max_segment_frames and not is_speech) or (len(speech_frames) >= int(max_segment_frames * 1.5))
            
            if speech_frames and (hit_hangover or hit_max_len):
                is_hard_cut = (not hit_hangover) and (len(speech_frames) >= int(max_segment_frames * 1.5))
                
                if is_hard_cut:
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
                    cushion_frames = max(1, 150 // cfg.frame_ms)
                    strip_count = max(0, silence_run - cushion_frames)
                    if strip_count > 0 and len(speech_frames) > strip_count:
                        audio_frames = speech_frames[:-strip_count]
                    else:
                        audio_frames = speech_frames
                    carryover_frames = []
                    carryover_probabilities = []

                if len(audio_frames) - silence_run >= min_speech_frames:
                    audio = np.concatenate(audio_frames).astype(np.float32)
                    segment_duration = len(audio_frames) * (cfg.frame_ms / 1000.0)
                    segments.append({
                        "audio": audio,
                        "start_ts": speech_start_ts,
                        "end_ts": speech_start_ts + segment_duration
                    })
                    
                if is_hard_cut:
                    speech_frames = list(carryover_frames)
                    speech_probabilities = list(carryover_probabilities)
                    silence_run = 0
                    speech_start_ts = t_offset - len(speech_frames) * (cfg.frame_ms / 1000.0)
                    preroll_buffer = []
                else:
                    preroll_buffer = speech_frames[-preroll_frames:] if len(speech_frames) >= preroll_frames else list(speech_frames)
                    speech_frames = []
                    speech_probabilities = []
                    silence_run = 0
                    speech_start_ts = None
                
            t_offset += cfg.frame_ms / 1000.0
            
        if speech_frames:
            actual_speech_len = len(speech_frames) - silence_run
            if actual_speech_len >= min_speech_frames:
                audio = np.concatenate(speech_frames).astype(np.float32)
                segments.append({
                    "audio": audio,
                    "start_ts": speech_start_ts,
                    "end_ts": t_offset
                })
                
        return segments

    # Run VAD on both full channels
    print("Post-processing: running VAD on full microphone channel...")
    mic_segs = get_speech_segments_sync(mic_audio, is_mic=True)
    print(f"Detected {len(mic_segs)} speech segments on microphone channel.")
    
    print("Post-processing: running VAD on full system loopback channel...")
    sys_segs = get_speech_segments_sync(sys_audio, is_mic=False)
    print(f"Detected {len(sys_segs)} speech segments on system loopback channel.")
    
    processed_segments = []
    
    # Transcribe Microphone segments (Speaker 1)
    for seg in mic_segs:
        try:
            # Segment RMS check
            rms = np.sqrt(np.mean(seg["audio"]**2)) if len(seg["audio"]) > 0 else 0.0
            if rms < 0.005:
                print(f"Post-processing: skipping quiet mic segment (RMS: {rms:.4f})")
                continue

            acquired = stt_lock.acquire(timeout=30.0)
            if not acquired:
                print("CRITICAL: STT lock acquisition timed out during post-processing mic segment — GPU thread stuck")
                continue
            try:
                text = clean_text(mic_engine.transcribe(seg["audio"], cfg.sample_rate))
            finally:
                stt_lock.release()
            if text:
                if is_whisper_hallucination(text, rms):
                    print(f"Post-processing: skipping suspected Whisper hallucination '{text}' (RMS: {rms:.4f}, channel: Speaker 1)")
                    continue

                processed_segments.append({
                    "speaker": "Speaker 1",
                    "start_ts": seg["start_ts"],
                    "end_ts": seg["end_ts"],
                    "text": text
                })
        except Exception as e:
            print(f"Error transcribing post-processed mic segment at {seg['start_ts']:.1f}s: {e}")
            
    # Transcribe System loopback segments (Speaker 2)
    for seg in sys_segs:
        try:
            # Segment RMS check
            rms = np.sqrt(np.mean(seg["audio"]**2)) if len(seg["audio"]) > 0 else 0.0
            if rms < 0.003:
                print(f"Post-processing: skipping quiet sys segment (RMS: {rms:.4f})")
                continue

            acquired = stt_lock.acquire(timeout=30.0)
            if not acquired:
                print("CRITICAL: STT lock acquisition timed out during post-processing sys segment — GPU thread stuck")
                continue
            try:
                text = clean_text(sys_engine.transcribe(seg["audio"], cfg.sample_rate))
            finally:
                stt_lock.release()
            if text:
                if is_whisper_hallucination(text, rms):
                    print(f"Post-processing: skipping suspected Whisper hallucination '{text}' (RMS: {rms:.4f}, channel: Speaker 2)")
                    continue

                processed_segments.append({
                    "speaker": "Speaker 2",
                    "start_ts": seg["start_ts"],
                    "end_ts": seg["end_ts"],
                    "text": text
                })
        except Exception as e:
            print(f"Error transcribing post-processed sys segment at {seg['start_ts']:.1f}s: {e}")
            
    # Sort chronologically by start time
    processed_segments.sort(key=lambda s: s["start_ts"])
    
    # Merge consecutive segments of the same speaker
    final_segments = []
    if processed_segments:
        current_seg = processed_segments[0]
        for s in processed_segments[1:]:
            # If same speaker AND close in time (gap < 3.0s)
            if s["speaker"] == current_seg["speaker"] and (s["start_ts"] - current_seg["end_ts"] < 3.0):
                current_seg["text"] = current_seg["text"] + " " + s["text"]
                current_seg["end_ts"] = s["end_ts"]
            else:
                final_segments.append(current_seg)
                current_seg = s
        final_segments.append(current_seg)
        
    return final_segments

class StopRequest(BaseModel):
    do_not_save: bool = False

@app.post("/api/stop")
def stop_recording(request: StopRequest = None):
    global mic_worker, sys_worker, recording_active
    global current_session_segments, current_session_start_time
    if not recording_active:
        return {"status": "not_recording"}
        
    print("Stopping recording...")
    if mic_worker:
        mic_worker.stop()
    if sys_worker:
        sys_worker.stop()
        
    recording_active = False
    
    # Prepare statistics
    stats = {
        "duration": getattr(mic_engine, "total_audio_duration", 0.0) + getattr(sys_engine, "total_audio_duration", 0.0),
        "mic": {
            "segments": mic_engine.total_segments,
            "duration": mic_engine.total_audio_duration,
            "inference_time": mic_engine.total_transcribe_time,
            "rtf": mic_engine.total_transcribe_time / mic_engine.total_audio_duration if mic_engine.total_audio_duration > 0 else 0.0
        },
        "sys": {
            "segments": sys_engine.total_segments,
            "duration": sys_engine.total_audio_duration,
            "inference_time": sys_engine.total_transcribe_time,
            "rtf": sys_engine.total_transcribe_time / sys_engine.total_audio_duration if sys_engine.total_audio_duration > 0 else 0.0
        }
    }
    
    # Save the meeting session if we transcribing anything
    global current_meeting_id
    meeting_id = current_meeting_id
    do_not_save_flag = request.do_not_save if request is not None else False
    if not do_not_save_flag and meeting_id:
        now = datetime.datetime.now()
        meeting_title = f"Meeting at {now.strftime('%I:%M %p')}"
        
        # Delete active_meeting.json as this session is now finalized
        active_meeting_file = os.path.join(cfg.history_dir, "active_meeting.json")
        if os.path.exists(active_meeting_file):
            try:
                os.remove(active_meeting_file)
            except Exception as e:
                print(f"Failed to remove active_meeting.json: {e}")
        
        # 1. Combine audio histories and save as stereo WAV
        import scipy.io.wavfile as wav
        mic_audio = np.concatenate(mic_worker.audio_history) if (mic_worker and mic_worker.audio_history) else np.empty(0, dtype=np.float32)
        sys_audio = np.concatenate(sys_worker.audio_history) if (sys_worker and sys_worker.audio_history) else np.empty(0, dtype=np.float32)
        
        max_len = max(len(mic_audio), len(sys_audio))
        # True wall-clock length of the saved recording, derived from the actual
        # audio samples (not the sum of live-transcribed segment durations below,
        # which excludes silence and undercounts the real file length - that
        # mismatch was causing the meeting JSON to report a much shorter
        # "duration" than the wav file actually is, e.g. 8.8s reported vs a
        # 20.3s real file).
        recorded_duration = max_len / cfg.sample_rate if max_len > 0 else 0.0
        if max_len > 0:
            if len(mic_audio) < max_len:
                mic_audio = np.pad(mic_audio, (0, max_len - len(mic_audio)))
            if len(sys_audio) < max_len:
                sys_audio = np.pad(sys_audio, (0, max_len - len(sys_audio)))
                
            stereo_audio = np.column_stack((mic_audio, sys_audio))
            wav_path = os.path.join(cfg.history_dir, f"{meeting_id}.wav")
            try:
                wav.write(wav_path, cfg.sample_rate, (np.clip(stereo_audio, -1.0, 1.0) * 32767).astype(np.int16))
                print(f"Saved meeting audio to {wav_path}")
            except Exception as e:
                print(f"Failed to save WAV audio: {e}")

        # 2. Reprocess the entire recorded audio (Full VAD & STT refinement)
        reprocessed_segments = reprocess_recording(
            mic_audio=mic_audio,
            sys_audio=sys_audio,
            cfg=cfg,
            mic_engine=mic_engine,
            sys_engine=sys_engine
        )
                        
        meeting_data = {
            "id": meeting_id,
            "title": meeting_title,
            "date": now.isoformat(),
            "duration": recorded_duration,
            "segments": reprocessed_segments,
            "stats": stats
        }
        
        file_path = os.path.join(cfg.history_dir, f"{meeting_id}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(meeting_data, f, ensure_ascii=False, indent=2)
            print(f"Saved session history to {file_path}")
        except Exception as e:
            print(f"Failed to save session history: {e}")
            
    print("Recording stopped.")
    return {"status": "stopped", "stats": stats, "meeting_id": meeting_id}

@app.get("/api/status")
def get_status():
    global mic_engine, is_loading_engine, engine_load_error
    
    model_name = cfg.model_size
    for m in MODELS_METADATA:
        if m["id"] == cfg.model_size:
            model_name = m["name"]
            break
            
    active_meeting_file = os.path.join(cfg.history_dir, "active_meeting.json")
    has_recoverable = False
    recoverable_data = None
    if os.path.exists(active_meeting_file):
        try:
            with open(active_meeting_file, "r", encoding="utf-8") as f:
                recoverable_data = json.load(f)
                has_recoverable = True
        except Exception:
            pass

    return {
        "recording": recording_active,
        "device": mic_engine.device if mic_engine else "Not Loaded",
        "model": cfg.model_size,
        "model_name": model_name,
        "loading": is_loading_engine,
        "error": engine_load_error,
        "stt_language": cfg.stt_language,
        "stt_initial_prompt": cfg.stt_initial_prompt,
        "stt_hotwords": cfg.stt_hotwords,
        "stt_prefix": cfg.stt_prefix,
        "has_recoverable": has_recoverable,
        "recoverable_meeting": recoverable_data
    }

@app.post("/api/recover")
def recover_meeting():
    global mic_worker, sys_worker, recording_active, out_queue
    global current_session_segments, current_session_start_time, current_meeting_id
    
    if recording_active:
        return {"status": "already_recording"}
        
    active_meeting_file = os.path.join(cfg.history_dir, "active_meeting.json")
    if not os.path.exists(active_meeting_file):
        raise HTTPException(status_code=400, detail="No active meeting to recover.")
        
    try:
        with open(active_meeting_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        meeting_id = data["id"]
        
        # Restore state
        current_meeting_id = meeting_id
        current_session_segments = data.get("segments", [])
        try:
            current_session_start_time = datetime.datetime.fromisoformat(data["date"])
        except Exception:
            current_session_start_time = datetime.datetime.now()
            
        # Clear the queue
        while not out_queue.empty():
            try:
                out_queue.get_nowait()
            except queue.Empty:
                break
                
        # Re-initialize workers
        echo_canceller = None
        if cfg.enable_aec:
            echo_canceller = WebRTCAcousticEchoCanceller(
                sample_rate=cfg.sample_rate,
                delay_ms=cfg.aec_delay_ms,
                enable_ns=cfg.aec_enable_ns,
                enable_agc=cfg.aec_enable_agc,
                ducking_threshold=cfg.aec_ducking_threshold,
                ref_threshold=cfg.aec_ref_threshold,
                correlation_threshold=cfg.aec_correlation_threshold,
                correlation_lag_search_ms=cfg.aec_correlation_lag_search_ms,
            )
            
        shared_state = {"last_sys_audio_active_time": 0.0}
        
        mic_worker = ChannelWorker(
            speaker_label="Speaker 1",
            capture=MicCapture(cfg.sample_rate, cfg.frame_ms, cfg.mic_device),
            segmenter=build_segmenter(cfg),
            engine=mic_engine,
            out_queue=out_queue,
            echo_canceller=echo_canceller,
            shared_state=shared_state,
            meeting_id=meeting_id
        )
        sys_worker = ChannelWorker(
            speaker_label="Speaker 2",
            capture=SystemAudioCapture(cfg.sample_rate, cfg.frame_ms, cfg.speaker_device),
            segmenter=build_segmenter(cfg),
            engine=sys_engine,
            out_queue=out_queue,
            echo_canceller=echo_canceller,
            shared_state=shared_state,
            meeting_id=meeting_id
        )
        
        # Reset stats
        mic_engine.total_segments = 0
        mic_engine.total_audio_duration = 0.0
        mic_engine.total_transcribe_time = 0.0
        
        sys_engine.total_segments = 0
        sys_engine.total_audio_duration = 0.0
        sys_engine.total_transcribe_time = 0.0
        
        # Start workers
        mic_worker.start()
        sys_worker.start()
        recording_active = True
        
        return {"status": "recovered", "meeting_id": meeting_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to recover session: {e}")

@app.post("/api/discard")
def discard_meeting():
    global cfg
    active_meeting_file = os.path.join(cfg.history_dir, "active_meeting.json")
    if os.path.exists(active_meeting_file):
        try:
            os.remove(active_meeting_file)
            return {"status": "discarded"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete active meeting: {e}")
    return {"status": "no_active_meeting"}

@app.get("/api/hardware-recommendation")
def get_recommendation():
    try:
        return get_model_recommendation()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config")
def get_config():
    global cfg
    return {
        "model_size": cfg.model_size,
        "engine_type": cfg.engine_type,
        "device": cfg.device,
        "compute_type": cfg.compute_type,
        "mic_device": cfg.mic_device,
        "speaker_device": cfg.speaker_device,
        "stt_language": cfg.stt_language,
        "stt_initial_prompt": cfg.stt_initial_prompt,
        "stt_hotwords": cfg.stt_hotwords,
        "stt_prefix": cfg.stt_prefix
    }

@app.post("/api/config")
def update_config(data: dict = Body(...)):
    global recording_active, is_loading_engine, cfg
    if recording_active:
        raise HTTPException(status_code=400, detail="Cannot change configuration while recording is active.")
        
    # Check if we are updating mic/speaker devices
    if "mic_device" in data:
        cfg.mic_device = data.get("mic_device")
    if "speaker_device" in data:
        cfg.speaker_device = data.get("speaker_device")
    if "stt_language" in data:
        cfg.stt_language = data.get("stt_language")
        if mic_engine and hasattr(mic_engine, "language"):
            mic_engine.language = cfg.stt_language
        if sys_engine and hasattr(sys_engine, "language"):
            sys_engine.language = cfg.stt_language
    if "stt_initial_prompt" in data:
        cfg.stt_initial_prompt = data.get("stt_initial_prompt")
        if mic_engine and hasattr(mic_engine, "initial_prompt"):
            mic_engine.initial_prompt = cfg.stt_initial_prompt
        if sys_engine and hasattr(sys_engine, "initial_prompt"):
            sys_engine.initial_prompt = cfg.stt_initial_prompt
    if "stt_hotwords" in data:
        cfg.stt_hotwords = data.get("stt_hotwords")
        if mic_engine and hasattr(mic_engine, "hotwords"):
            mic_engine.hotwords = cfg.stt_hotwords
        if sys_engine and hasattr(sys_engine, "hotwords"):
            sys_engine.hotwords = cfg.stt_hotwords
    if "stt_prefix" in data:
        cfg.stt_prefix = data.get("stt_prefix")
        if mic_engine and hasattr(mic_engine, "prefix"):
            mic_engine.prefix = cfg.stt_prefix
        if sys_engine and hasattr(sys_engine, "prefix"):
            sys_engine.prefix = cfg.stt_prefix
        
    model_size = data.get("model_size")
    if model_size and model_size != cfg.model_size:
        if is_loading_engine:
            raise HTTPException(status_code=400, detail="Model is already loading in the background.")
        # Start the background engine loading thread
        threading.Thread(target=bg_load_engines, args=(model_size,), daemon=True).start()
        return {"status": "loading"}
        
    return {"status": "success"}

@app.get("/api/audio-devices")
def get_audio_devices():
    import soundcard as sc
    try:
        # soundcard fetches list of microphone and speaker devices
        mics = [{"id": m.id, "name": m.name} for m in sc.all_microphones()]
        speakers = [{"id": s.id, "name": s.name} for s in sc.all_speakers()]
        return {
            "mics": mics,
            "speakers": speakers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch audio devices: {str(e)}")

@app.post("/api/test-sound")
def test_sound(data: dict = Body(...)):
    import soundcard as sc
    import numpy as np
    
    speaker_device = data.get("speaker_device")
    try:
        speaker = sc.get_speaker(speaker_device) if speaker_device else sc.default_speaker()
        
        # Generate a beautiful 440Hz sine wave chime (0.4s) fading out exponentially
        sample_rate = 44100
        duration = 0.4
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        envelope = np.exp(-6 * t)
        chime = np.sin(2 * np.pi * 440 * t) * envelope * 0.25
        
        # Stereo audio array
        stereo_chime = np.column_stack((chime, chime))
        
        # Play via soundcard
        with speaker.player(samplerate=sample_rate) as p:
            p.play(stereo_chime)
            
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to play test sound: {str(e)}")

@app.get("/api/models")
def get_models():
    try:
        rec = get_model_recommendation()
        rec_model = rec.get("recommended_model")
        
        result = []
        for m in MODELS_METADATA:
            model_id = m["id"]
            downloaded = check_model_downloaded(model_id)
            downloading = model_id in downloading_models
            is_rec = model_id == rec_model
            
            result.append({
                "id": model_id,
                "name": m["name"],
                "size": m["size"],
                "speed": m["speed"],
                "accuracy": m["accuracy"],
                "downloaded": downloaded,
                "downloading": downloading,
                "progress": download_progress.get(model_id, 0) if downloading else 0,
                "is_recommended": is_rec
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/download")
def download_model(data: dict = Body(...)):
    global downloading_models
    model_id = data.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required.")
        
    valid_ids = [m["id"] for m in MODELS_METADATA]
    if model_id not in valid_ids:
        raise HTTPException(status_code=400, detail=f"Invalid model_id '{model_id}'.")
        
    if check_model_downloaded(model_id):
        return {"status": "already_downloaded"}
        
    if model_id in downloading_models:
        return {"status": "downloading"}
        
    downloading_models.add(model_id)
    threading.Thread(target=download_model_worker, args=(model_id,), daemon=True).start()
    return {"status": "started"}

@app.post("/api/models/cancel")
def cancel_model(data: dict = Body(...)):
    global downloading_models, cancelled_downloads
    model_id = data.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required.")
        
    if model_id in downloading_models:
        cancelled_downloads.add(model_id)
        return {"status": "cancelling"}
    return {"status": "not_downloading"}

@app.post("/api/models/delete")
def delete_model(data: dict = Body(...)):
    global downloading_models
    model_id = data.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required.")
        
    if model_id in downloading_models:
        raise HTTPException(status_code=400, detail="Cannot delete a model while it is downloading.")
        
    try:
        delete_model_files(model_id)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_sockets.append(websocket)
    try:
        while True:
            # We must await something to keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_sockets.remove(websocket)

def is_echo_duplicate(event_text, event_start, event_end, event_speaker, segments):
    if event_speaker != "Speaker 1":
        return False
        
    import string
    def get_words(t):
        return set(t.lower().translate(str.maketrans("", "", string.punctuation)).split())
        
    w_me = get_words(event_text)
    if not w_me:
        return False
        
    for s in segments[-15:]:
        if s["speaker"] == "Speaker 2":
            time_close = abs(event_start - s["start_ts"]) <= 4.0 or abs(event_end - s["end_ts"]) <= 4.0
            if time_close:
                w_sys = get_words(s["text"])
                if not w_sys:
                    continue
                intersection = w_me.intersection(w_sys)
                union = w_me.union(w_sys)
                similarity = len(intersection) / len(union) if union else 0
                if similarity >= 0.65:
                    return True
    return False

# Background task to drain transcription queue, store them in memory, and broadcast to websockets
async def queue_listener():
    global out_queue, active_sockets, current_session_segments
    
    while True:
        try:
            # Drain new events from out_queue and process/broadcast immediately
            while not out_queue.empty():
                try:
                    event = out_queue.get_nowait()
                except queue.Empty:
                    break
                
                speaker = event.speaker
                text = event.text
                start_ts = event.start_ts
                end_ts = event.end_ts
                is_final = getattr(event, "is_final", True)
                
                # Check for post-transcription semantic deduplication safety net
                if is_final:
                    if is_echo_duplicate(text, start_ts, end_ts, speaker, current_session_segments):
                        print(f"[AEC Safety Net] Suppressed duplicate mic transcript event: '{text}'")
                        continue
 
                # Prepare segment for broadcast
                segment_data = {
                    "speaker": speaker,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "text": text,
                    "is_final": is_final
                }
                
                # Keep in active session segment store if final
                if is_final:
                    current_session_segments.append(segment_data)
                
                # Broadcast to all websocket clients
                for ws in list(active_sockets):
                    try:
                        await ws.send_json(segment_data)
                    except Exception:
                        if ws in active_sockets:
                            active_sockets.remove(ws)
                            
        except Exception as e:
            print(f"Error in queue_listener background loop: {e}")
            import traceback
            traceback.print_exc()
            
        await asyncio.sleep(0.05)


def print_audio_devices():
    try:
        import soundcard as sc
        print("\n" + "="*45)
        print("           DETECTED AUDIO DEVICES            ")
        print("="*45)
        print("Microphones (Capture Devices):")
        for idx, m in enumerate(sc.all_microphones()):
            print(f"  - {m.name}")
        print("\nSpeakers (Playback Devices):")
        for idx, s in enumerate(sc.all_speakers()):
            print(f"  - {s.name}")
        try:
            print(f"\n[Default Microphone] {sc.default_microphone().name}")
        except Exception:
            print("\n[Default Microphone] NONE")
        try:
            print(f"[Default Speaker]    {sc.default_speaker().name}")
        except Exception:
            print("[Default Speaker]    NONE")
        print("="*45 + "\n")
    except Exception as e:
        print(f"Failed to query audio devices: {e}")

# Start queue listener background loop when app starts
@app.on_event("startup")
def start_listener():
    print_audio_devices()
    loop = asyncio.get_event_loop()
    loop.create_task(queue_listener())

# --- History APIs ---
@app.get("/api/history")
def get_history():
    history_list = []
    if not os.path.exists(cfg.history_dir):
        return history_list
        
    for filename in os.listdir(cfg.history_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(cfg.history_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    history_list.append({
                        "id": data.get("id"),
                        "title": data.get("title"),
                        "date": data.get("date"),
                        "duration": data.get("duration"),
                        "num_segments": len(data.get("segments", [])),
                        "full_text": " ".join([seg.get("text", "") for seg in data.get("segments", [])])
                    })
            except Exception as e:
                print(f"Error reading history file {filename}: {e}")
                
    # Sort by date descending
    history_list.sort(key=lambda x: x.get("date", ""), reverse=True)
    return history_list

@app.get("/api/history/{meeting_id}")
def get_meeting_details(meeting_id: str):
    file_path = os.path.join(cfg.history_dir, f"{meeting_id}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read meeting details: {str(e)}")

@app.put("/api/history/{meeting_id}")
def rename_meeting(meeting_id: str, request: RenameRequest):
    file_path = os.path.join(cfg.history_dir, f"{meeting_id}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["title"] = request.title
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rename meeting: {str(e)}")

@app.delete("/api/history/{meeting_id}")
def delete_meeting(meeting_id: str):
    file_path = os.path.join(cfg.history_dir, f"{meeting_id}.json")
    wav_path = os.path.join(cfg.history_dir, f"{meeting_id}.wav")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        os.remove(file_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete meeting: {str(e)}")

from fastapi.responses import FileResponse

@app.get("/api/history/{meeting_id}/audio")
def get_meeting_audio(meeting_id: str):
    file_path = os.path.join(cfg.history_dir, f"{meeting_id}.wav")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/wav")

# Mount static/compiled frontend files
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))

if os.path.exists(frontend_dist):
    print(f"Mounting compiled frontend from: {frontend_dist}")
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
elif os.path.exists(static_dir):
    print(f"Mounting legacy static files from: {static_dir}")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    print("Warning: Neither frontend dist nor static directory was found.")

def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info", access_log=False)

if __name__ == "__main__":
    # Force stdout/stderr to use UTF-8 encoding on Windows to avoid console print crashes.
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    # Check if run with --webview to run pywebview standalone
    if "--webview" in sys.argv:
        # Start FastAPI in a background thread
        t = threading.Thread(target=start_server, daemon=True)
        t.start()
        
        # Create pywebview desktop window
        webview.create_window(
            title="Local Transcribe AI",
            url="http://127.0.0.1:8000",
            width=1024,
            height=768,
            resizable=True
        )
        webview.start()
    else:
        # Default behavior: run only the backend server (used for Tauri sidecar/dev)
        start_server()
