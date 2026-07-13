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
    def __init__(self, model_size: str = "distil-large-v3", remote_url: str = "http://127.0.0.1:8001/transcribe", language: str | None = None, initial_prompt: str | None = None):
        self.model_size = model_size
        self.remote_url = remote_url
        self.language = language
        self.initial_prompt = initial_prompt
        self.device = f"Remote GPU ({self.remote_url})"
        self.total_transcribe_time = 0.0
        self.total_segments = 0
        self.total_audio_duration = 0.0
        # Use a persistent session to benefit from connection pooling / TCP keep-alive
        self.session = requests.Session()

    def transcribe(self, audio: np.ndarray, samplerate: int) -> str:
        if samplerate != 16000:
            raise ValueError("Whisper expects 16kHz audio.")
            
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
        
        # Call the remote endpoint using the persistent session
        try:
            # We pass the model_size to run on the server (in case the server runs multiple models, or to log it)
            response = self.session.post(
                self.remote_url,
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={
                    "model_size": self.model_size,
                    "language": self.language or "",
                    "initial_prompt": self.initial_prompt or ""
                },
                timeout=90.0  # Increased to allow the remote GPU to download/load the model on first call
            )
            if response.status_code == 200:
                text = response.json().get("text", "").strip()
            else:
                text = f"[Error: Remote server status {response.status_code}]"
        except requests.exceptions.Timeout:
            text = f"[Error: Connection timed out. The remote server is likely downloading or warming up the model '{self.model_size}' on the GPU. Please try speaking again in a moment.]"
        except requests.exceptions.RequestException as e:
            text = f"[Error: Connection to remote GPU failed: {str(e)}]"
            
        duration = time.perf_counter() - start_t
        self.total_transcribe_time += duration
        self.total_segments += 1
        self.total_audio_duration += len(audio) / samplerate
        
        return text
