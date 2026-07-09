import os
import sys
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
from audio.vad_segmenter import VADSegmenter
from audio.echo_cancellation import WebRTCAcousticEchoCanceller
from pipeline.channel_worker import ChannelWorker, TranscriptEvent
from pipeline.transcript_normalizer import format_line
from stt.moonshine_dml_engine import MoonshineDirectMLEngine
from stt.faster_whisper_engine import FasterWhisperEngine
from stt.remote_engine import RemoteSTTEngine
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
        "id": "distil-small.en",
        "name": "Distil Small EN",
        "size": "164 MB",
        "speed": "very-fast",
        "accuracy": "high acc"
    },
    {
        "id": "distil-medium.en",
        "name": "Distil Medium EN",
        "size": "383 MB",
        "speed": "fast",
        "accuracy": "very-high acc"
    },
    {
        "id": "distil-large-v3",
        "name": "Distil Large v3",
        "size": "731 MB",
        "speed": "medium",
        "accuracy": "very-high acc"
    },
    {
        "id": "distil-large-v2",
        "name": "Distil Large v2",
        "size": "731 MB",
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
        "id": "remote/distil-large-v3",
        "name": "Remote GPU (RTX 5090) - Distil Large v3",
        "size": "731 MB",
        "speed": "blazing-fast",
        "accuracy": "excellent acc"
    },
    {
        "id": "remote/large-v3-turbo",
        "name": "Remote GPU (RTX 5090) - Whisper Large v3 Turbo",
        "size": "1.6 GB",
        "speed": "blazing-fast",
        "accuracy": "excellent acc"
    }
]


from pathlib import Path
import shutil

downloading_models = set()

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
    else:
        models_dir = Path(cfg.model_download_root)
        repo_name = f"models--Systran--faster-whisper-{model_id}"
        repo_dir = models_dir / repo_name
        if repo_dir.exists():
            for p in repo_dir.glob("**/model.bin"):
                return True
        return False

def download_model_worker(model_id: str):
    global downloading_models
    try:
        print(f"Starting download for model '{model_id}'...")
        if model_id.startswith("moonshine/"):
            # Instantiate MoonshineDirectMLEngine on CPU to download weights safely
            MoonshineDirectMLEngine(
                model_size=model_id,
                device="cpu",
                compute_type="float",
                download_root=cfg.model_download_root
            )
        else:
            from faster_whisper import WhisperModel
            WhisperModel(
                model_id,
                device="cpu",
                compute_type="int8",
                download_root=cfg.model_download_root
            )
        print(f"Finished downloading model '{model_id}' successfully.")
    except Exception as e:
        print(f"Error downloading model '{model_id}': {e}")
    finally:
        if model_id in downloading_models:
            downloading_models.remove(model_id)

def delete_model_files(model_id: str):
    if model_id.startswith("remote/"):
        return
    hf_cache_dir = Path(os.environ.get("HF_HUB_CACHE", "E:/Local transcribe/local-transcribe/models"))
    if model_id.startswith("moonshine/"):
        for name in ["models--useful-sensors--moonshine", "models--UsefulSensors--moonshine"]:
            repo_dir = hf_cache_dir / name
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
    else:
        models_dir = Path(cfg.model_download_root)
        repo_name = f"models--Systran--faster-whisper-{model_id}"
        repo_dir = models_dir / repo_name
        if repo_dir.exists():
            shutil.rmtree(repo_dir)

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
        new_mic_engine = RemoteSTTEngine(
            model_size=actual_model,
            remote_url=cfg.remote_url
        )
        new_sys_engine = RemoteSTTEngine(
            model_size=actual_model,
            remote_url=cfg.remote_url
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
        new_mic_engine = FasterWhisperEngine(
            model_size,
            device,
            compute_type,
            download_root=cfg.model_download_root,
        )
        new_sys_engine = FasterWhisperEngine(
            model_size,
            device,
            compute_type,
            download_root=cfg.model_download_root,
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
    )

@app.on_event("startup")
def startup_event():
    global cfg
    print("Initializing STT engines in background...")
    import threading
    threading.Thread(target=bg_load_engines, args=(cfg.model_size,), daemon=True).start()

@app.post("/api/start")
def start_recording():
    global mic_worker, sys_worker, recording_active, out_queue
    global current_session_segments, current_session_start_time
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
            
    # Initialize echo canceller if enabled
    echo_canceller = None
    if cfg.enable_aec:
        echo_canceller = WebRTCAcousticEchoCanceller(
            sample_rate=cfg.sample_rate,
            delay_ms=cfg.aec_delay_ms,
            enable_ns=cfg.aec_enable_ns,
            enable_agc=cfg.aec_enable_agc,
            ducking_threshold=cfg.aec_ducking_threshold,
            ref_threshold=cfg.aec_ref_threshold
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
    )
    sys_worker = ChannelWorker(
        speaker_label="Speaker 2",
        capture=SystemAudioCapture(cfg.sample_rate, cfg.frame_ms, cfg.speaker_device),
        segmenter=build_segmenter(cfg),
        engine=sys_engine,
        out_queue=out_queue,
        echo_canceller=echo_canceller,
        shared_state=shared_state,
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
    import webrtcvad
    from pipeline.channel_worker import stt_lock
    from pipeline.transcript_normalizer import clean_text
    
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
            
        vad = webrtcvad.Vad(cfg.vad_aggressiveness)
        # Use a larger silence hangover (800ms) for post-processing to group continuous speech
        post_hangover_ms = 800
        hangover_frames = max(1, post_hangover_ms // cfg.frame_ms)
        min_speech_frames = max(1, cfg.min_speech_ms // cfg.frame_ms)
        # Use a larger max segment limit (25.0s) to keep full sentence structures together
        post_max_segment_s = 25.0
        max_segment_frames = int(post_max_segment_s * 1000 / cfg.frame_ms)
        
        speech_frames = []
        silence_run = 0
        speech_start_ts = None
        
        segments = []
        t_offset = 0.0
        
        for idx in range(0, len(audio_array), frame_samples):
            chunk = audio_array[idx : idx + frame_samples]
            if len(chunk) < frame_samples:
                chunk = np.pad(chunk, (0, frame_samples - len(chunk)))
                
            rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0.0
            
            clipped = np.clip(chunk, -1.0, 1.0)
            pcm_bytes = (clipped * 32767).astype(np.int16).tobytes()
            
            is_speech = vad.is_speech(pcm_bytes, cfg.sample_rate)
            
            # Dynamic threshold masking for microphone channel to suppress room echo leakage
            current_rms_threshold = 0.012 if is_mic else 0.008
            if is_mic and sys_active_times:
                # Check if system loopback was active in the last 1.0s
                was_sys_active = any(t_offset - 1.0 <= st <= t_offset for st in sys_active_times)
                if was_sys_active:
                    current_rms_threshold = 0.035
                    
            if rms < current_rms_threshold:
                is_speech = False
                
            if is_speech:
                if not speech_frames:
                    speech_start_ts = t_offset
                speech_frames.append(chunk)
                silence_run = 0
            elif speech_frames:
                speech_frames.append(chunk)
                silence_run += 1
                
            hit_hangover = speech_frames and silence_run >= hangover_frames
            hit_max_len = (len(speech_frames) >= max_segment_frames and not is_speech) or (len(speech_frames) >= int(max_segment_frames * 1.5))
            
            if speech_frames and (hit_hangover or hit_max_len):
                if len(speech_frames) - silence_run >= min_speech_frames:
                    audio = np.concatenate(speech_frames).astype(np.float32)
                    segments.append({
                        "audio": audio,
                        "start_ts": speech_start_ts,
                        "end_ts": t_offset
                    })
                speech_frames = []
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
            if rms < 0.012:
                print(f"Post-processing: skipping quiet mic segment (RMS: {rms:.4f})")
                continue

            with stt_lock:
                text = clean_text(mic_engine.transcribe(seg["audio"], cfg.sample_rate))
            if text:
                # Suspected Whisper hallucination check on low energy
                cleaned_lower = text.lower().strip().replace("’", "'").translate(str.maketrans("", "", ".,?!"))
                if cleaned_lower in ["thank you", "thank you so much", "i don't know", "you", "yeah", "yes", "oh", "bye"]:
                    if rms < 0.025:
                        print(f"Post-processing: skipping suspected Whisper hallucination '{text}' (RMS: {rms:.4f})")
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
            with stt_lock:
                text = clean_text(sys_engine.transcribe(seg["audio"], cfg.sample_rate))
            if text:
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
    meeting_id = None
    do_not_save_flag = request.do_not_save if request is not None else False
    if not do_not_save_flag and (current_session_segments or stats["duration"] > 0):
        now = datetime.datetime.now()
        meeting_id = f"meeting_{now.strftime('%Y%m%d_%H%M%S')}"
        meeting_title = f"Meeting at {now.strftime('%I:%M %p')}"
        
        # 1. Combine audio histories and save as stereo WAV
        import scipy.io.wavfile as wav
        mic_audio = np.concatenate(mic_worker.audio_history) if (mic_worker and mic_worker.audio_history) else np.empty(0, dtype=np.float32)
        sys_audio = np.concatenate(sys_worker.audio_history) if (sys_worker and sys_worker.audio_history) else np.empty(0, dtype=np.float32)
        
        max_len = max(len(mic_audio), len(sys_audio))
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
            "duration": stats["duration"],
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
            
    return {
        "recording": recording_active,
        "device": mic_engine.device if mic_engine else "Not Loaded",
        "model": cfg.model_size,
        "model_name": model_name,
        "loading": is_loading_engine,
        "error": engine_load_error
    }

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
        "speaker_device": cfg.speaker_device
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
        
    model_size = data.get("model_size")
    if model_size:
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
                
                # Check for post-transcription semantic deduplication safety net
                if is_echo_duplicate(text, start_ts, end_ts, speaker, current_session_segments):
                    print(f"[AEC Safety Net] Suppressed duplicate mic transcript event: '{text}'")
                    continue

                # Prepare segment for broadcast
                segment_data = {
                    "speaker": speaker,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "text": text
                }
                
                # Keep in active session segment store
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
