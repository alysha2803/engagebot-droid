"""
Phase 4 validation: live noise-level meter from the microphone.

Standalone, like multi_face_test.py / body_language_test.py were for their
phases -- no DB writes, no video, just a terminal meter to sanity-check
that RMS/dBFS classification (audio_cues.py) reacts sensibly: quiet when
the room is silent, moderate during normal talking, loud when you raise
your voice or make noise close to the mic.

Captures from the laptop's default mic for now (same "test with what's on
hand first" approach used for the webcam before the Pi camera was wired
up) -- swapping to the Pi's mic over the network is a later step once this
logic itself is confirmed to make sense.

Usage:
    python noise_test.py
"""
import os
import sys
import time

import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_cues import rms_dbfs, classify_noise

SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.5
BAR_WIDTH = 40

_LEVEL_BAR_FLOOR = -60.0   # dBFS mapped to an empty bar
_LEVEL_BAR_CEIL  = 0.0     # dBFS mapped to a full bar


def _bar(dbfs):
    frac = (dbfs - _LEVEL_BAR_FLOOR) / (_LEVEL_BAR_CEIL - _LEVEL_BAR_FLOOR)
    frac = max(0.0, min(1.0, frac))
    filled = int(BAR_WIDTH * frac)
    return "#" * filled + "-" * (BAR_WIDTH - filled)


def main():
    print(f"Listening on default mic ({SAMPLE_RATE} Hz, {CHUNK_SECONDS}s chunks). Ctrl+C to stop.\n")
    chunk_frames = int(SAMPLE_RATE * CHUNK_SECONDS)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
            while True:
                audio, _ = stream.read(chunk_frames)
                pcm = audio[:, 0]
                dbfs = rms_dbfs(pcm)
                level, score = classify_noise(dbfs)
                print(f"\r{dbfs:6.1f} dBFS  [{_bar(dbfs)}]  {level:9s} score={score:.2f}   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == "__main__":
    main()
