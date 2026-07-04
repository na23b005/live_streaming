import queue
import signal
import sys
import time

# Import config first so Hugging Face environment variables are set before other modules load.
from config import Config

from audio.capture import MicCapture, SystemAudioCapture
from audio.vad_segmenter import VADSegmenter
from pipeline.channel_worker import ChannelWorker, TranscriptEvent
from pipeline.transcript_normalizer import format_line
from stt.faster_whisper_engine import FasterWhisperEngine

# Suppress the soundcard "data discontinuity in recording" warnings.
# We do this here after importing audio.capture because soundcard internally calls
# warnings.simplefilter('always', SoundcardRuntimeWarning) which overrides earlier filters.
import warnings
warnings.filterwarnings("ignore", message=".*data discontinuity in recording.*")


def build_segmenter(cfg: Config) -> VADSegmenter:
    return VADSegmenter(
        samplerate=cfg.sample_rate,
        frame_ms=cfg.frame_ms,
        aggressiveness=cfg.vad_aggressiveness,
        silence_hangover_ms=cfg.silence_hangover_ms,
        min_speech_ms=cfg.min_speech_ms,
        max_segment_s=cfg.max_segment_s,
    )


def main() -> None:
    cfg = Config()
    if len(sys.argv) > 1:
        cfg.model_size = sys.argv[1]
    out_queue: "queue.Queue[TranscriptEvent]" = queue.Queue()
    session_start = time.perf_counter()

    print(f"Loading Whisper model '{cfg.model_size}' (device={cfg.device})...")
    # Two separate model instances so each channel can transcribe independently
    # without waiting on the other. Uses more RAM; if that's tight, see the
    # README note on sharing a single instance across both channels.
    mic_engine = FasterWhisperEngine(
        cfg.model_size,
        cfg.device,
        cfg.compute_type,
        download_root=cfg.model_download_root,
    )
    sys_engine = FasterWhisperEngine(
        cfg.model_size,
        cfg.device,
        cfg.compute_type,
        download_root=cfg.model_download_root,
    )
    print(f"Model loaded. Running on: {mic_engine.device}")

    mic_worker = ChannelWorker(
        speaker_label="Me",
        capture=MicCapture(cfg.sample_rate, cfg.frame_ms, cfg.mic_device),
        segmenter=build_segmenter(cfg),
        engine=mic_engine,
        out_queue=out_queue,
    )
    sys_worker = ChannelWorker(
        speaker_label="Speaker 1",
        capture=SystemAudioCapture(cfg.sample_rate, cfg.frame_ms, cfg.speaker_device),
        segmenter=build_segmenter(cfg),
        engine=sys_engine,
        out_queue=out_queue,
    )

    workers = [mic_worker, sys_worker]
    for w in workers:
        w.start()

    def shutdown(*_args):
        print("\nStopping...")
        for w in workers:
            w.stop()
        
        # Drain any remaining transcript events from the output queue
        while not out_queue.empty():
            try:
                event = out_queue.get_nowait()
                line = format_line(event.speaker, event.start_ts, event.text)
                print(line)
                with open(cfg.transcript_log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(line + "\n")
            except queue.Empty:
                break
        
        session_duration = time.perf_counter() - session_start
        print("\n=== Transcription Session Statistics ===")
        print(f"Session Duration: {session_duration:.2f} seconds")
        for name, engine in [("Microphone (Me)", mic_engine), ("System Audio (Speaker 1)", sys_engine)]:
            print(f"\n{name}:")
            print(f"  Segments Transcribed: {engine.total_segments}")
            if engine.total_segments > 0:
                print(f"  Total Audio Duration: {engine.total_audio_duration:.2f} seconds")
                print(f"  Total Whisper Inference Time: {engine.total_transcribe_time:.2f} seconds")
                avg_time = engine.total_transcribe_time / engine.total_segments
                print(f"  Average Time per Segment: {avg_time:.2f} seconds")
                rt_ratio = engine.total_transcribe_time / engine.total_audio_duration if engine.total_audio_duration > 0 else 0
                print(f"  Real-time Factor: {rt_ratio:.3f} (lower is faster, < 1.0 is faster than real-time)")
            else:
                print("  No audio segments were transcribed.")
        print("========================================\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    print("Listening... (Ctrl+C to stop)\n")
    with open(cfg.transcript_log_path, "a", encoding="utf-8") as log_file:
        while True:
            try:
                event = out_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            line = format_line(event.speaker, event.start_ts, event.text)
            print(line)
            log_file.write(line + "\n")
            log_file.flush()


if __name__ == "__main__":
    main()
