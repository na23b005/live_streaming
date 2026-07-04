# local-transcribe

A minimal, fully local live transcription + speaker-separation tool, built the
same way Natively (Cluely) does it: capture microphone and system-output
audio as two *separate hardware channels* instead of running a neural
diarization model, then transcribe each channel independently with Whisper
and merge the results into one live, timestamped transcript.

```
You speak into the mic        -> "Me"
Anything playing out speakers -> "Speaker 1"
```

No audio ever leaves your machine.

---

## 1. Tech stack, and why

| Piece | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | Fastest path to a correct, minimal implementation. Rust (like the reference) is faster and lower-latency, but Python + PortAudio/soundcard + faster-whisper gets you the same behavior with far less code. Swap to Rust later if you need the lock-free/real-time guarantees the reference native module has. |
| Mic capture | [`soundcard`](https://github.com/bastibe/SoundCard) | Cross-platform (PortAudio-free, pure ctypes) mic capture. |
| System audio capture | `soundcard` loopback mode | Same library gives you WASAPI loopback on Windows and the PulseAudio/Pipewire monitor source on Linux through one API — this is the trickiest part of the whole pipeline and having one library cover both OSes avoids two separate code paths. |
| Voice activity detection | `webrtcvad` | Tiny, fast, no model file, good enough to detect speech vs. silence for endpointing. This plays the role of Natively's `VadProcessor`. |
| Speech-to-text | [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) | The most efficient Whisper implementation for CPU inference — ~4-6x faster than plain Whisper with `int8` quantization, negligible accuracy loss on small models. This plays the role of Natively's `whisperWorker.ts` + ONNX Runtime. |
| Diarization | None (by design) | Same trick as the reference: hardware channel separation. Mic → `me`, loopback → `speaker_1`. This is `TranscriptNormalizer.ts` / `SpeakerLabelService.ts`'s canonical-id scheme, reimplemented in `pipeline/transcript_normalizer.py`. |

### What's *not* in this MVP (on purpose, per your priorities)
- **No GPU acceleration wired up yet.** You said hardware/quantization tuning is optional and you'd try small models first — `tiny.en`/`base.en` run comfortably in real time on any modern CPU, including with an AMD GPU idle in the machine. See §4 for the real GPU path when you want it.
- **No meeting summary / LLM step.** You said you don't need it now. `SpeakerLabelService.applyLabels()`'s job (renaming `speaker_1` → a real name before an LLM call) is the natural place to add it back later — see §5.
- **No partial/interim ("live-typing") transcription.** Natively uses LocalAgreement-2 for that. This MVP transcribes a full utterance once it detects a pause (see §3), which is simpler and still feels live (typically 0.5-1.5s after someone stops talking), but won't show words appearing mid-sentence.

---

## 2. Project structure

```
local-transcribe/
├── config.py                        # all tunable parameters in one place
├── main.py                          # entry point: wires everything together
├── requirements.txt
├── audio/
│   ├── capture.py                   # MicCapture, SystemAudioCapture (soundcard)
│   └── vad_segmenter.py             # WebRTC VAD -> speech segment boundaries
├── stt/
│   ├── base.py                      # STTEngine interface (swap backends here)
│   └── faster_whisper_engine.py     # faster-whisper implementation
└── pipeline/
    ├── channel_worker.py            # capture -> VAD -> STT per channel, threaded
    └── transcript_normalizer.py     # filler-word/repetition cleanup + formatting
```

Data flow for **one** channel (there are two of these running concurrently,
one for mic, one for system audio):

```
AudioCapture (thread) --chunks--> VADSegmenter (thread) --segments--> STTEngine --text--> shared out_queue
```

`main.py` just drains the shared queue and prints/logs whatever arrives from
either channel, in arrival order.

---

## 3. Installation

### Windows

```powershell
# 1. Python 3.10+ (from python.org or Microsoft Store)
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```

`soundcard`'s loopback mode uses WASAPI directly on Windows, so no extra
drivers are needed — it captures the default playback device automatically.

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# webrtcvad needs a C compiler to build from source on some distros:
sudo apt install build-essential python3-dev   # Debian/Ubuntu

python main.py
```

`soundcard` picks up the PulseAudio/Pipewire "monitor" source for your
default output automatically — no manual `pactl` setup needed in most
distros. If system-audio capture comes back silent, run `pactl list sources`
and confirm a `*.monitor` source exists and is not suspended.

### macOS

`soundcard` cannot do loopback capture on macOS without a virtual audio
device (there's no OS-level API for it, same reason the Rust reference uses
ScreenCaptureKit/CoreAudio directly instead of a generic library). Install
[BlackHole](https://github.com/ExistentialAudio/BlackHole) (2ch), set it as a
multi-output alongside your speakers, and point `speaker_device` in
`config.py` at the BlackHole device name.

### First run

The first run downloads the Whisper model (`base.en` by default, ~150MB) from
Hugging Face into your local cache — this needs internet once, then works
fully offline. If it's slow on your CPU, drop to `tiny.en` in `config.py`.

---

## 4. GPU acceleration path (for your AMD RX 6650, later)

This matters enough to be precise about: **faster-whisper's backend
(CTranslate2) only accelerates on NVIDIA CUDA or CPU.** There is no ROCm or
DirectML build of it. So today, on an AMD card, this pipeline runs on CPU —
which was your explicit fallback plan, and is genuinely fine for
`tiny.en`/`base.en`.

When you want real AMD acceleration (and something that's actually
vendor-agnostic — AMD, NVIDIA, Intel, from the same binary), the
architecture is already set up for it:

- `stt/base.py` defines `STTEngine` as the only contract `ChannelWorker` talks to.
- Write a new `stt/whispercpp_vulkan_engine.py` implementing that same interface,
  backed by [whisper.cpp](https://github.com/ggerganov/whisper.cpp) compiled with
  `GGML_VULKAN=1`. Vulkan compute shaders run on AMD/NVIDIA/Intel GPUs on
  both Windows and Linux — it's the same reason game engines target Vulkan
  for cross-vendor GPU support. This is a closer cross-platform match to
  what the reference project does for Windows (`dml` execution provider)
  than trying to get ROCm working, since ROCm's officially supported card
  list is narrow and doesn't reliably include consumer cards like the
  RX 6650.
- Swap `FasterWhisperEngine(...)` for `WhisperCppVulkanEngine(...)` in
  `main.py`. Nothing else changes.

The reference doc's per-module mixed-precision (`fp32` encoder / `q8`
decoder) trick is also worth carrying over once you're on whisper.cpp —
GGML supports per-tensor quantization the same way.

---

## 5. Extending this later

- **Renaming speakers / multi-participant labels**: reintroduce something
  like `SpeakerLabelService.ts` — a small `dict[str, str]` mapping
  `"speaker_1"` -> a real name, applied when formatting output.
- **Summaries / action items**: pipe the merged transcript (or
  `transcript.log`) into a local LLM via [Ollama](https://ollama.com), same
  as `OllamaManager.ts` does — this is a separate, optional post-processing
  step and doesn't touch the live pipeline above.
- **Partial/interim results**: add a LocalAgreement-2 loop around the VAD
  segmenter (re-run STT on a growing buffer every ~1.5s, keep only the
  longest common prefix across consecutive runs) if you want words to
  appear while someone is still talking rather than after they pause.
- **True diarization** (multiple people on the *same* mic, e.g. an in-person
  meeting with one laptop mic): this hardware-separation trick can't help
  there — you'd need an actual embedding-based diarizer (pyannote or
  similar), which is exactly why the reference project's own docs call this
  approach an MVP rather than full diarization.

---

## 6. Tuning quick-reference (`config.py`)

| If you see... | Change |
|---|---|
| Transcription lags noticeably behind speech | Drop `model_size` to `"tiny.en"` |
| Segments cut off mid-sentence | Increase `silence_hangover_ms` (e.g. 800-1000) |
| Long pause before you see any text | Decrease `silence_hangover_ms`, but expect more mid-sentence splits |
| Short coughs/clicks show up as garbage segments | Increase `min_speech_ms` or `vad_aggressiveness` |
| System audio channel is silent | Confirm loopback source (see Linux section), or set `speaker_device` explicitly |
