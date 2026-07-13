import io
import os
os.environ["HF_TOKEN"] = "hf_8*****"
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
        
        load_path = model_size
        if "/" in model_size:
            from pathlib import Path
            safe_name = model_size.replace("/", "--")
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            out_dir = models_dir / f"models--{safe_name}-ct2"
            
            if not out_dir.exists():
                print(f"Custom model '{model_size}' not found. Downloading and converting to CTranslate2...")
                raw_dir = models_dir / f"models--{safe_name}-raw"
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=model_size, local_dir=raw_dir)
                
                import sys
                import subprocess
                subprocess.run([
                    sys.executable, "-m", "ctranslate2.converters.transformers",
                    "--model", str(raw_dir),
                    "--output_dir", str(out_dir),
                    "--copy_files", "tokenizer.json", "preprocessor_config.json",
                    "--quantization", "float16" # float16 on remote GPU
                ], check=True)
                
                import shutil
                shutil.rmtree(raw_dir)
            load_path = str(out_dir)
            
        models_cache[model_size] = WhisperModel(
            load_path,
            device="cuda",
            compute_type="float16"
        )
        print(f"Model '{model_size}' loaded successfully.")
    return models_cache[model_size]

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...), 
    model_size: str = Form("large-v3-turbo"),
    language: str = Form(None),
    initial_prompt: str = Form(None)
):
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided.")
        
    try:
        audio_bytes = await file.read()
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"
        
        # Load the model dynamically (cached)
        model = get_model(model_size)
        
        # Transcribe audio segment
        b_size = 2 if language == "en" else 1
        segments, info = model.transcribe(
            audio_file, 
            beam_size=b_size, 
            language=language if language else None, 
            initial_prompt=initial_prompt if initial_prompt else None, 
            vad_filter=True
        )
        
        texts = []
        for seg in segments:
            # Relaxed thresholds to prevent trailing accented words from being skipped
            if seg.no_speech_prob > 0.85 or seg.avg_logprob < -1.5:
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
        get_model("large-v3-turbo")
    except Exception as e:
        print(f"Warning: Failed to warm up model on startup: {e}. It will load on first request.")
        
    uvicorn.run(app, host=args.host, port=args.port)
