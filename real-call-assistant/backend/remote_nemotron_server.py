import io
import wave
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn
import torch
import os
import tempfile
import soundfile as sf
import traceback
import json
import re
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
            
        manifest_name = None
        try:
            supported_langs = {'en-US', 'en', 'en-GB', 'enGB', 'es-ES', 'esES', 'es-US', 'es', 'zh-CN', 'zh-ZH'}
            cleaned_lang = language if language in supported_langs else "en-GB"
            asr_model.set_inference_prompt(cleaned_lang)
            
            # Build a temporary manifest file to pass the language metadata down to Lhotse cuts
            duration = len(audio_np) / 16000.0
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as manifest_file:
                manifest_name = manifest_file.name
                json.dump({
                    "audio_filepath": tmp_name,
                    "duration": duration,
                    "text": "",
                    "lang": cleaned_lang,
                    "language": cleaned_lang,
                    "source_lang": cleaned_lang
                }, manifest_file)
                manifest_file.write("\n")
                
            with torch.no_grad():
                results = asr_model.transcribe(manifest_name, target_lang=cleaned_lang)
                print(f"[NemotronServer] raw results: {results} (type={type(results)})")
                
                text = ""
                if isinstance(results, dict):
                    text = results.get("text", results.get("pred_text", ""))
                elif isinstance(results, list) and results:
                    first_res = results[0]
                    print(f"[NemotronServer] first_res: {first_res} (type={type(first_res)})")
                    if isinstance(first_res, str):
                        text = first_res
                    elif isinstance(first_res, dict):
                        text = first_res.get("text", first_res.get("pred_text", ""))
                    elif hasattr(first_res, "text"):
                        text = first_res.text
                    else:
                        text = str(first_res)
                else:
                    text = str(results) if results is not None else ""
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            if manifest_name and os.path.exists(manifest_name):
                os.remove(manifest_name)
                
        # Clean language tags (e.g. <en-US>, <en-GB>, <en>) from the transcription
        text = re.sub(r"\s*<[a-zA-Z]{2,3}(?:-[a-zA-Z]{2,4})?>\s*", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
                
        return {"text": text}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
