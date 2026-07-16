import time
import numpy as np
from .base import STTEngine

class MoonshineDirectMLEngine(STTEngine):
    def __init__(self, model_size: str = "moonshine/base", device: str = "dml", compute_type: str = "float", download_root: str | None = None):
        import onnxruntime as ort
        from moonshine_onnx.model import MoonshineOnnxModel
        from moonshine_onnx.transcribe import load_tokenizer
        
        # Determine execution providers for the encoder based on device setting
        if device == "dml":
            self.providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            self.providers = ["CPUExecutionProvider"]
            
        # We subclass MoonshineOnnxModel to load encoder on DirectML/CPU, and decoder strictly on CPU.
        # This bypasses AMD GPU DirectX 12 driver compilation bugs in the autoregressive decoder loop.
        class CustomMoonshineOnnxModel(MoonshineOnnxModel):
            def __init__(self, model_name, precision, providers_list):
                import onnxruntime
                model_name = model_name.split("/")[-1]
                
                # Download weights using Moonshine's internal method
                encoder, decoder = self._load_weights_from_hf_hub(model_name, precision)
                
                # Load encoder with DirectML (GPU) with CPU fallback
                self.encoder = onnxruntime.InferenceSession(encoder, providers=providers_list)
                
                # Load decoder strictly with CPU to prevent AMD driver shader compilation bugs
                self.decoder = onnxruntime.InferenceSession(decoder, providers=["CPUExecutionProvider"])
                
                self.encoder_input_names = [x.name for x in self.encoder.get_inputs()]
                self.decoder_input_names = [x.name for x in self.decoder.get_inputs()]
                
                if "tiny" in model_name:
                    self.num_layers = 6
                    self.num_key_value_heads = 8
                    self.head_dim = 36
                elif "base" in model_name:
                    self.num_layers = 8
                    self.num_key_value_heads = 8
                    self.head_dim = 52
                else:
                    raise ValueError(f'Unknown model "{model_name}"')
                
                self.decoder_start_token_id = 1
                self.eos_token_id = 2
                
        self.model = CustomMoonshineOnnxModel(model_size, compute_type, self.providers)
        self.tokenizer = load_tokenizer()
        
        # Format the active device description
        enc_provider = self.model.encoder.get_providers()[0]
        self.device = f"DirectML GPU + CPU Decoder ({enc_provider})" if enc_provider == "DmlExecutionProvider" else "CPU"
        
        self.total_transcribe_time = 0.0
        self.total_segments = 0
        self.total_audio_duration = 0.0

        # Warm up model to compile kernels and warm caches (pre-empt first-utterance latency)
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        _ = self.transcribe(dummy_audio, 16000)

    def transcribe(self, audio: np.ndarray, samplerate: int, meeting_id: str | None = None) -> str:
        if samplerate != 16000:
            raise ValueError("Moonshine expects 16kHz audio.")
            
        # Root mean square (RMS) threshold to filter out silence/ambient hum
        rms = np.sqrt(np.mean(audio**2)) if len(audio) > 0 else 0.0
        if rms < 0.006:
            return ""
            
        start_t = time.perf_counter()
        
        # Audio needs shape [1, samples]
        audio_input = audio[None, :].astype(np.float32)
        
        # Check size constraints of Moonshine (expects >0.1s audio)
        num_seconds = audio_input.size / 16000
        if num_seconds <= 0.1:
            return ""
            
        # Run inference
        tokens = self.model.generate(audio_input)
        text = self.tokenizer.decode_batch(tokens)[0].strip()
        
        duration = time.perf_counter() - start_t
        self.total_transcribe_time += duration
        self.total_segments += 1
        self.total_audio_duration += num_seconds
        
        return text
