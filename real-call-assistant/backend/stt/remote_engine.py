import io
import time
import wave
import numpy as np
import requests
from .base import STTEngine

class RemoteSTTEngine(STTEngine):
    """
    STTEngine that sends audio segments to a remote server (e.g., RTX 5090 via Tailscale)
    running our FastAPI-based faster-whisper server.
    """
    def __init__(self, model_size: str = "distil-large-v3", remote_url: str = "http://127.0.0.1:8001/transcribe", language: str | None = None, initial_prompt: str | None = None, hotwords: str | None = None, prefix: str | None = None):
        self.model_size = model_size
        self.remote_url = remote_url
        self.language = language
        self.initial_prompt = initial_prompt
        self.hotwords = hotwords
        self.prefix = prefix
        self.device = f"Remote GPU ({self.remote_url})"
        self.total_transcribe_time = 0.0
        self.total_segments = 0
        self.total_audio_duration = 0.0
        # Use a persistent session to benefit from connection pooling / TCP keep-alive
        self.session = requests.Session()

    def transcribe(self, audio: np.ndarray, samplerate: int, meeting_id: str | None = None) -> str:
        if samplerate != 16000:
            raise ValueError("Whisper expects 16kHz audio.")
            
        # Root mean square (RMS) threshold to filter out silence/ambient hum
        rms = np.sqrt(np.mean(audio**2)) if len(audio) > 0 else 0.0
        if rms < 0.006:
            return ""
            
        start_t = time.perf_counter()
        
        # Convert the float32 array in range [-1.0, 1.0] to 16-bit PCM WAV
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(samplerate)
            
            # Avoid overflow clipping
            clipped = np.clip(audio, -1.0, 1.0)
            pcm_audio = (clipped * 32767).astype(np.int16)
            wav_file.writeframes(pcm_audio.tobytes())
            
        wav_bytes = wav_io.getvalue()
        
        # Call the remote endpoint using the persistent session with retry logic
        resp_json = None
        for attempt in range(2):
            try:
                response = self.session.post(
                    self.remote_url,
                    files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                    data={
                        "model_size": self.model_size,
                        "language": self.language or "",
                        "initial_prompt": self.initial_prompt or "",
                        "hotwords": self.hotwords or "",
                        "prefix": self.prefix or ""
                    },
                    timeout=90.0  # Increased to allow the remote GPU to download/load the model on first call
                )
                if response.status_code == 200:
                    resp_json = response.json()
                    break
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt == 0:
                    print(f"[RemoteSTT] Connection warning: {e}. Recreating session and retrying...")
                    self.session = requests.Session()
                    time.sleep(0.5)
                    continue
                else:
                    print(f"[RemoteSTT] Connection failed after retry: {e}")
                    raise
                    
        if resp_json is None:
            return "[Error: Connection to remote GPU failed (aborted).]"
            
        text = resp_json.get("text", "").strip()
        segments_data = resp_json.get("segments", [])
                
        # Check for low confidence segments to save locally on the client
        if meeting_id and segments_data:
            import scipy.io.wavfile as wav
            import os
            from config import HISTORY_DIR
            
            for seg in segments_data:
                avg_logprob = seg.get("avg_logprob", 0.0)
                if avg_logprob < -1.0:
                    start_s = seg.get("start", 0.0)
                    end_s = seg.get("end", 0.0)
                    
                    start_idx = int(start_s * samplerate)
                    end_idx = int(end_s * samplerate)
                    seg_audio = audio[start_idx:end_idx]
                    if len(seg_audio) > 0:
                        debug_dir = os.path.join(HISTORY_DIR, f"{meeting_id}_low_conf")
                        os.makedirs(debug_dir, exist_ok=True)
                        filename = f"seg_{start_s:.2f}_{end_s:.2f}_prob_{avg_logprob:.2f}.wav"
                        wav_path = os.path.join(debug_dir, filename)
                        try:
                            wav.write(wav_path, samplerate, (np.clip(seg_audio, -1.0, 1.0) * 32767).astype(np.int16))
                            print(f"[LowConf-Remote] Saved low-confidence segment to {wav_path}")
                        except Exception as ex:
                            print(f"[LowConf-Remote] Failed to save WAV segment: {ex}")
            
        duration = time.perf_counter() - start_t
        self.total_transcribe_time += duration
        self.total_segments += 1
        self.total_audio_duration += len(audio) / samplerate
        
        return text
