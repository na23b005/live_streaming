import io
import time
import wave
import numpy as np
import requests
from .base import STTEngine

class NemotronRemoteEngine(STTEngine):
    """
    STTEngine that sends audio segments to a remote server running our FastAPI-based
    NVIDIA Nemotron 3.5 ASR server (typically on RTX 5090 via Tailscale).
    """
    def __init__(self, model_size: str = "nvidia-nemotron-3.5", remote_url: str = "http://127.0.0.1:8001/transcribe", language: str | None = None):
        self.model_size = model_size
        # If remote_url is the whisper url (e.g. ending in 8000 or /transcribe with Whisper endpoint),
        # we adjust it to point to our Nemotron server on port 8001
        self.remote_url = remote_url
        if "8000" in self.remote_url:
            self.remote_url = self.remote_url.replace("8000", "8001")
        if not self.remote_url.endswith("/transcribe"):
            # Ensure it ends with /transcribe
            if self.remote_url.endswith("/"):
                self.remote_url += "transcribe"
            else:
                self.remote_url += "/transcribe"
                
        # Hardcode language as en-GB for Indian English
        self.language = "en-US"
            
        self.device = f"Remote GPU Nemotron ({self.remote_url})"
        self.total_transcribe_time = 0.0
        self.total_segments = 0
        self.total_audio_duration = 0.0
        self.session = requests.Session()

    def transcribe(self, audio: np.ndarray, samplerate: int, meeting_id: str | None = None) -> str:
        if samplerate != 16000:
            raise ValueError("ASR model expects 16kHz audio.")
            
        # Root mean square (RMS) threshold to filter out silence/ambient hum
        rms = np.sqrt(np.mean(audio**2)) if len(audio) > 0 else 0.0
        if rms < 0.005:  # Matches our lowered quiet threshold
            return ""
            
        start_t = time.perf_counter()
        
        # Convert float32 array [-1.0, 1.0] to 16-bit PCM WAV bytes
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(samplerate)
            
            clipped = np.clip(audio, -1.0, 1.0)
            pcm_audio = (clipped * 32767).astype(np.int16)
            wav_file.writeframes(pcm_audio.tobytes())
            
        wav_bytes = wav_io.getvalue()
        
        # Call the remote Nemotron FastAPI server
        resp_json = None
        for attempt in range(2):
            try:
                response = self.session.post(
                    self.remote_url,
                    files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                    data={"language": self.language},
                    timeout=30.0
                )
                if response.status_code == 200:
                    resp_json = response.json()
                    break
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt == 0:
                    print(f"[RemoteNemotron] Connection warning: {e}. Retrying...")
                    self.session = requests.Session()
                    time.sleep(0.5)
                    continue
                else:
                    print(f"[RemoteNemotron] Connection failed: {e}")
                    raise
                    
        if resp_json is None:
            return "[Error: Connection to remote Nemotron GPU failed.]"
            
        text = resp_json.get("text", "").strip()
        
        duration = time.perf_counter() - start_t
        self.total_transcribe_time += duration
        self.total_segments += 1
        self.total_audio_duration += len(audio) / samplerate
        
        return text
