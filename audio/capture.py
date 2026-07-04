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
        self.out_queue: "queue.Queue[np.ndarray]" = queue.Queue()
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
        with self._get_recorder() as rec:
            while not self._stop.is_set():
                data = rec.record(numframes=self.chunk_samples)  # shape (n, channels)
                mono = data.mean(axis=1) if data.ndim > 1 else data
                self.out_queue.put(mono.astype(np.float32))


class MicCapture(AudioCapture):
    """Physical microphone input. Canonical speaker: 'Me'."""

    def _get_recorder(self):
        mic = sc.get_microphone(self.device_name) if self.device_name else sc.default_microphone()
        return mic.recorder(samplerate=self.samplerate)


class SystemAudioCapture(AudioCapture):
    """Loopback of whatever is playing out of the speakers (remote call
    audio, video, music, etc). Canonical speaker: 'Speaker 1'.

    Note (same limitation the reference Rust code documents): this follows
    the OS *default* output device. If the meeting app is routed to a
    non-default device, point speaker_device at it explicitly.
    """

    def _get_recorder(self):
        speaker = sc.get_speaker(self.device_name) if self.device_name else sc.default_speaker()
        loopback_mic = sc.get_microphone(speaker.id, include_loopback=True)
        return loopback_mic.recorder(samplerate=self.samplerate)
