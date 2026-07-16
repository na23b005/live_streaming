"""
Hardware-separated audio capture.

This is the core of "diarization" here, same as Natively: instead of running a
neural diarization model on a single mixed stream, we capture two physically
distinct channels and never mix them:

  - MicCapture           -> physical microphone input   -> always "Me"
  - SystemAudioCapture   -> speaker/output loopback      -> always "Speaker 1"

soundcard is used because it exposes loopback recording with the same API on
both Windows (WASAPI loopback) and Linux (PulseAudio/Pipewire monitor source),
so this file needs no per-OS branches. macOS does not support loopback without
a virtual audio driver (e.g. BlackHole) - see README.
"""

import time
import queue
import threading

import numpy as np
import soundcard as sc



class AudioCapture:
    """Continuously records from a device and pushes fixed-size, mono,
    float32 chunks onto self.out_queue. Runs in its own thread so a slow
    consumer never causes the underlying recorder to drop or block audio.
    """

    def __init__(self, samplerate: int, chunk_ms: int = 30, device_name: str | None = None):
        self.samplerate = samplerate
        self.chunk_samples = int(samplerate * chunk_ms / 1000)
        self.device_name = device_name
        self.out_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=100)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _get_recorder(self):
        raise NotImplementedError

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        consecutive_flatlines = 0
        max_flatlines_before_restart = 100  # ~3 seconds at 30/32ms chunks
        
        while not self._stop.is_set():
            try:
                with self._get_recorder() as rec:
                    consecutive_flatlines = 0
                    while not self._stop.is_set():
                        data = rec.record(numframes=self.chunk_samples)  # shape (n, channels)
                        mono = data.mean(axis=1) if data.ndim > 1 else data
                        
                        # Monitor health: zero-variance check for physical microphone
                        if isinstance(self, MicCapture):
                            if np.var(mono) == 0.0:
                                consecutive_flatlines += 1
                                if consecutive_flatlines >= max_flatlines_before_restart:
                                    print(f"[AudioCapture] Warning: {self.__name__ if hasattr(self, '__name__') else self.__class__.__name__} detected flatline zero variance for {consecutive_flatlines} frames. Re-initializing WASAPI recorder...")
                                    break
                            else:
                                consecutive_flatlines = 0
                                
                        item = (time.perf_counter(), mono.astype(np.float32))
                        try:
                            self.out_queue.put(item, timeout=2.0)
                        except queue.Full:
                            try:
                                self.out_queue.get_nowait()
                                self.out_queue.put_nowait(item)
                            except (queue.Empty, queue.Full):
                                pass
            except Exception as e:
                print(f"[AudioCapture] Error in {self.__class__.__name__} capture loop: {e}")
                if self._stop.is_set():
                    break
                time.sleep(0.5)  # wait before retrying to open the recorder
            else:
                # If we broke out of the inner loop due to flatline detection, wait briefly before recreating recorder
                if not self._stop.is_set():
                    time.sleep(0.1)


class MicCapture(AudioCapture):
    """Physical microphone input. Canonical speaker: 'Me'."""

    def _get_recorder(self):
        try:
            mic = sc.get_microphone(self.device_name) if self.device_name else sc.default_microphone()
        except Exception as e:
            print(f"[AudioCapture] Warning: Requested microphone '{self.device_name}' not available ({e}). Falling back to default.")
            mic = sc.default_microphone()
            
        blocksize = int(self.samplerate * 60 / 1000)  # 60ms buffer helps prevent WASAPI underruns
        return mic.recorder(samplerate=self.samplerate, blocksize=blocksize)


class SystemAudioCapture(AudioCapture):
    """Loopback of whatever is playing out of the speakers (remote call
    audio, video, music, etc). Canonical speaker: 'Speaker 1'.

    Note (same limitation the reference Rust code documents): this follows
    the OS *default* output device. If the meeting app is routed to a
    non-default device, point speaker_device at it explicitly.
    """

    def __init__(self, samplerate: int, chunk_ms: int = 30, device_name: str | None = None):
        super().__init__(samplerate, chunk_ms, device_name)
        self._silence_stop = threading.Event()
        self._silence_thread = None

    def start(self) -> None:
        super().start()
        self._silence_stop.clear()
        self._silence_thread = threading.Thread(
            target=self._play_silence_loop,
            name="SilencePlayer",
            daemon=True
        )
        self._silence_thread.start()

    def stop(self) -> None:
        self._silence_stop.set()
        if self._silence_thread:
            self._silence_thread.join(timeout=2)
        super().stop()

    def _play_silence_loop(self) -> None:
        import time

        # Self-healing: if the playback loop below throws mid-meeting (device
        # unplugged, output switched, format change), reopen the speaker and
        # keep going instead of letting the thread die and silently losing
        # the keep-alive protection for the rest of the meeting.
        while not self._silence_stop.is_set():
            try:
                # Play to default speaker or specified device
                speaker = sc.get_speaker(self.device_name) if self.device_name else sc.default_speaker()
            except Exception:
                try:
                    speaker = sc.default_speaker()
                except Exception:
                    time.sleep(0.5)
                    continue

            try:
                # Keep a tiny 10ms silent loop running to keep WASAPI driver active when paused
                samplerate = 48000
                chunk_size = 480
                silence = np.zeros((chunk_size, 2), dtype=np.float32)

                with speaker.player(samplerate=samplerate, blocksize=chunk_size) as p:
                    while not self._silence_stop.is_set():
                        p.play(silence)
                        time.sleep(0.005)
            except Exception as e:
                print(f"[SystemAudioCapture] Background silence player loop stopped: {e}. Retrying...")
                if not self._silence_stop.is_set():
                    time.sleep(0.5)

    def _get_recorder(self):
        try:
            speaker = sc.get_speaker(self.device_name) if self.device_name else sc.default_speaker()
        except Exception as e:
            print(f"[AudioCapture] Warning: Requested speaker '{self.device_name}' not available ({e}). Falling back to default.")
            speaker = sc.default_speaker()
            
        try:
            loopback_mic = sc.get_microphone(speaker.id, include_loopback=True)
        except Exception as e:
            print(f"[AudioCapture] Warning: Loopback matching speaker ID '{speaker.id}' not found ({e}). Trying speaker name '{speaker.name}'...")
            try:
                loopback_mic = sc.get_microphone(speaker.name, include_loopback=True)
            except Exception as e2:
                print(f"[AudioCapture] Warning: Loopback matching name '{speaker.name}' failed ({e2}). Attempting default speaker loopback.")
                try:
                    default_spk = sc.default_speaker()
                    loopback_mic = sc.get_microphone(default_spk.id, include_loopback=True)
                except Exception as e3:
                    print(f"[AudioCapture] Warning: Default speaker loopback failed ({e3}). Scanning for any available loopback device...")
                    # Fallback to the first available loopback microphone on the system
                    all_mics = sc.all_microphones(include_loopback=True)
                    loopback_mic = None
                    # Pick first loopback that isn't the physical mic array
                    for m in all_mics:
                        if m.id != "{0.0.1.00000000}.{0962d707-71ab-4445-925e-c5cf8d45c802}":
                            if m.id.startswith("{0.0.0.00000000}") or "loopback" in m.name.lower() or "monitor" in m.name.lower():
                                loopback_mic = m
                                break
                    if not loopback_mic and all_mics:
                        loopback_mic = all_mics[0]
                    if not loopback_mic:
                        raise RuntimeError("No loopback or audio capture devices found on the system.")
                
        blocksize = int(self.samplerate * 60 / 1000)  # 60ms buffer helps prevent WASAPI underruns
        return loopback_mic.recorder(samplerate=self.samplerate, blocksize=blocksize)
        return loopback_mic.recorder(samplerate=self.samplerate, blocksize=blocksize)
