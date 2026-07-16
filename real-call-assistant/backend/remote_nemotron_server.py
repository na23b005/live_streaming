import io
import wave
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn
import torch
import os
import tempfile
import soundfile as sf

app = FastAPI(title="Remote NVIDIA Nemotron 3.5 ASR Server")

# Load model globally on CUDA
print("Loading Nemotron 3.5 ASR model on CUDA...")
import nemo.collections.asr as nemo_asr
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the cache-aware streaming model (0.6B)
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/nemotron-3.5-asr-streaming-0.6b"
).to(device)

# The model requires prompt keys
asr_model.set_inference_prompt("en-US")
asr_model.eval()

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("en-US")
):
    try:
        audio_bytes = await file.read()
        
        # Read WAV bytes
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            if framerate != 16000:
                raise HTTPException(status_code=400, detail="Only 16kHz audio supported.")
                
            content = wav_file.readframes(n_frames)
            
            # Convert to float32
            if sampwidth == 2:
                audio_np = np.frombuffer(content, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                raise HTTPException(status_code=400, detail="Only 16-bit PCM WAV supported.")
                
        # Write to temporary file for NeMo's path-based transcription
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_name = tmp.name
            sf.write(tmp_name, audio_np, 16000, subtype='PCM_16')
            
        try:
            # Set language prompt dynamically (e.g. en-US, es-ES, etc.)
            asr_model.set_inference_prompt(language)
            with torch.no_grad():
                results = asr_model.transcribe([tmp_name])
                text = results[0] if results else ""
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
                
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
