# Nexus AI — Local Live Transcription & Diarization

Nexus AI is a fully local, hardware-separated live transcription and speaker diarization desktop application. It captures physical microphone input and system speaker loopback audio as distinct channels, transcribing each independently and merging them in real-time into a unified, timestamped transcript.

No audio data ever leaves your machine, ensuring 100% privacy.

---

## Key Features

- **Hardware-Separated Diarization**: Instead of running resource-heavy neural diarization models, the app captures microphone input ("Me") and speaker loopback ("Speaker 1") as separate audio devices.
- **Acoustic Echo Cancellation (AEC)**: A real-time pipeline filter cross-checks microphone inputs against system audio using a sliding alignment window and `difflib.SequenceMatcher` to suppress loopback audio leaking into physical mics (when not using headphones).
- **Multiple Local STT Engines**:
  - **Faster-Whisper**: High-speed Whisper engine quantized to `int8` for fast CPU execution.
  - **Moonshine (ONNX + DirectML)**: A lightweight, low-latency engine running GPU-accelerated inference on AMD, NVIDIA, and Intel graphics cards using DirectML.
- **Async Engine Loading**: The STT models load on background threads to ensure instant backend startup.
- **Draggable Frameless UI**: Built with Tauri v2 and React, presenting a premium custom title bar and a dark mode interface.
- **Meeting Management**: Save transcripts locally, rename sessions, copy transcripts to the clipboard, export to TXT, or delete history.

---

## Project Structure

```
local-transcribe/
├── backend/
│   ├── app.py                      # FastAPI server, background loader, WebSocket router
│   ├── config.py                   # App configuration & HuggingFace models directory redirection
│   ├── requirements.txt            # Python dependencies (soundcard, webrtcvad, faster-whisper, onnxruntime-directml)
│   ├── models/                     # [NEW LOCATION] Downloaded model weights (HuggingFace cache, gitignored)
│   ├── history/                    # [NEW LOCATION] Saved local meeting transcript JSONs (gitignored)
│   ├── audio/
│   │   ├── capture.py              # soundcard loopback & microphone audio recording threads
│   │   └── vad_segmenter.py        # WebRTC Voice Activity Detection for speech segmentation
│   ├── pipeline/
│   │   ├── channel_worker.py       # Threaded capture -> VAD -> STT loops per hardware channel
│   │   └── transcript_normalizer.py # Formatting, repetition cleanup, and text filtering
│   └── stt/
│       ├── base.py                 # Abstract STTEngine interface
│       ├── faster_whisper_engine.py# CTranslate2 / Faster-Whisper CPU engine
│       └── moonshine_dml_engine.py # UsefulSensors Moonshine ONNX + DirectML engine
│
└── frontend/
    ├── package.json
    ├── tauri.conf.json             # Tauri v2 window and system permissions config
    ├── src/
    │   ├── App.tsx                 # Core application view manager & background sync polling
    │   ├── components/             # Dashboard, MeetingDetail, SettingsModal, LiveTranscribeOverlay
    │   └── index.css               # Main visual styles (dark mode, glassmorphism scrollable cards)
    └── src-tauri/                  # Rust Tauri v2 wrapper
```

---

## Setup & Running

### 1. Requirements
- **OS**: Windows 10/11
- **Python**: Version 3.10 or 3.11
- **Node.js**: LTS version (18+)
- **Rust**: Version 1.77+ (Required to compile the Tauri native framework wrapper. Download via [rustup.rs](https://rustup.rs/))

---

### 2. Backend Setup
1. Open a terminal and navigate to the `backend/` directory:
   ```bash
   cd backend
   ```
2. Set up a virtual environment and activate it (the local `venv` folder is located at the root of the workspace, two levels up from the backend directory):
   ```powershell
   # Windows PowerShell
   ..\..\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   python app.py
   ```
   *The server runs locally on `http://127.0.0.1:8000` (access logs are disabled for terminal cleanliness).*

---

### 3. Frontend Setup (Tauri Dev Server)
1. Open a new terminal window and navigate to the `frontend/` directory:
   ```bash
   cd frontend
   ```
2. Install npm packages:
   ```bash
   npm install
   ```
3. Run the application in desktop dev mode:
   ```bash
   npm run tauri dev
   ```

---

### 4. Compiling Production Binaries
To bundle the frontend assets and compile the native Rust Tauri executable:
```bash
cd frontend
npm run tauri build
```
The compiled, standalone executable will be located in:
`frontend/src-tauri/target/release/local-transcribe.exe`

---

## Future Improvements

1. **Acoustic Echo Cancellation (current system works perfectly only if the meeting attendand wears earphone)**
2. **Dashboard Search Functionality**
3. **Real-Time Call Suggestions & Follow-ups**
4. **Migrate Backend to Rust**
