import os
import sys
import asyncio
import queue
import threading
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Import config first so Hugging Face environment variables are set before other modules load.
from config import Config
from audio.capture import MicCapture, SystemAudioCapture
from audio.vad_segmenter import VADSegmenter
from pipeline.channel_worker import ChannelWorker, TranscriptEvent
from pipeline.transcript_normalizer import format_line
from stt.moonshine_dml_engine import MoonshineDirectMLEngine
import webview

app = FastAPI()

# Global state
cfg = Config()
mic_engine = None
sys_engine = None
mic_worker = None
sys_worker = None
out_queue = queue.Queue()
recording_active = False
active_sockets = []

def build_segmenter(cfg: Config) -> VADSegmenter:
    return VADSegmenter(
        samplerate=cfg.sample_rate,
        frame_ms=cfg.frame_ms,
        aggressiveness=cfg.vad_aggressiveness,
        silence_hangover_ms=cfg.silence_hangover_ms,
        min_speech_ms=cfg.min_speech_ms,
        max_segment_s=cfg.max_segment_s,
    )

@app.on_event("startup")
def startup_event():
    global mic_engine, sys_engine, cfg
    print("Loading Moonshine STT engines...")
    # Load engines on startup so it's ready when the user clicks record
    mic_engine = MoonshineDirectMLEngine(
        cfg.model_size,
        cfg.device,
        compute_type=cfg.compute_type,
        download_root=cfg.model_download_root,
    )
    sys_engine = MoonshineDirectMLEngine(
        cfg.model_size,
        cfg.device,
        compute_type=cfg.compute_type,
        download_root=cfg.model_download_root,
    )
    print(f"STT engines loaded successfully. Device: {mic_engine.device}")

@app.post("/api/start")
def start_recording():
    global mic_worker, sys_worker, recording_active, out_queue
    if recording_active:
        return {"status": "already_recording"}
        
    # Clear the queue
    while not out_queue.empty():
        try:
            out_queue.get_nowait()
        except queue.Empty:
            break
            
    # Re-initialize workers
    mic_worker = ChannelWorker(
        speaker_label="Me",
        capture=MicCapture(cfg.sample_rate, cfg.frame_ms, cfg.mic_device),
        segmenter=build_segmenter(cfg),
        engine=mic_engine,
        out_queue=out_queue,
    )
    sys_worker = ChannelWorker(
        speaker_label="Speaker 1",
        capture=SystemAudioCapture(cfg.sample_rate, cfg.frame_ms, cfg.speaker_device),
        segmenter=build_segmenter(cfg),
        engine=sys_engine,
        out_queue=out_queue,
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

@app.post("/api/stop")
def stop_recording():
    global mic_worker, sys_worker, recording_active
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
    print("Recording stopped.")
    return {"status": "stopped", "stats": stats}

@app.get("/api/status")
def get_status():
    return {
        "recording": recording_active,
        "device": mic_engine.device if mic_engine else "Not Loaded",
        "model": cfg.model_size
    }

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

# Background task to drain transcription queue and broadcast to websockets
async def queue_listener():
    global out_queue, active_sockets
    while True:
        if not out_queue.empty():
            try:
                event = out_queue.get_nowait()
                # Broadcast to all websocket clients
                for ws in list(active_sockets):
                    try:
                        await ws.send_json({
                            "speaker": event.speaker,
                            "start_ts": event.start_ts,
                            "end_ts": event.end_ts,
                            "text": event.text
                        })
                    except Exception:
                        if ws in active_sockets:
                            active_sockets.remove(ws)
            except queue.Empty:
                pass
        await asyncio.sleep(0.05)

# Start queue listener background loop when app starts
@app.on_event("startup")
def start_listener():
    loop = asyncio.get_event_loop()
    loop.create_task(queue_listener())

# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

if __name__ == "__main__":
    # Force stdout/stderr to use UTF-8 encoding on Windows to avoid console print crashes.
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

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
