import io
import os
import argparse
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import uvicorn

app = FastAPI(title="Nexus AI - Remote GPU STT Server (RTX 5090)")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to cache loaded models on GPU
models_cache = {}

def get_model(model_size: str) -> WhisperModel:
    if model_size not in models_cache:
        print(f"Loading WhisperModel '{model_size}' onto RTX 5090 (device=cuda, compute_type=float16)...")
        # RTX 5090 features high-speed float16 tensor cores.
        models_cache[model_size] = WhisperModel(
            model_size,
            device="cuda",
            compute_type="float16"
        )
        print(f"Model '{model_size}' loaded successfully.")
    return models_cache[model_size]

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), model_size: str = Form("distil-large-v3")):
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided.")
        
    try:
        audio_bytes = await file.read()
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"
        
        # Load the model dynamically (cached)
        model = get_model(model_size)
        
        # Transcribe audio segment
        segments, info = model.transcribe(audio_file, beam_size=1, vad_filter=False)
        
        texts = []
        for seg in segments:
            if seg.no_speech_prob > 0.65 or seg.avg_logprob < -1.0:
                continue
            texts.append(seg.text.strip())
        text = " ".join(texts).strip()
        
        return {"text": text}
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    import torch
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown/No CUDA"
    return {
        "status": "ok",
        "gpu": gpu_name,
        "cuda_available": torch.cuda.is_available(),
        "loaded_models": list(models_cache.keys())
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to run the server on")
    args = parser.parse_args()
    
    # Warm up default model on startup
    try:
        get_model("distil-large-v3")
    except Exception as e:
        print(f"Warning: Failed to warm up model on startup: {e}. It will load on first request.")
        
    uvicorn.run(app, host=args.host, port=args.port)
