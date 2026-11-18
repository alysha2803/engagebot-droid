"""
Validation: live noise-level meter reading from the PI's mic over the
network (pi_stream_server.py's /audio.pcm), instead of the laptop's own
mic like noise_test.py does. Run this once after setting up the Pi's mic
to confirm the stream actually works and sounds reasonable, before
trusting it inside the full classroom_session.py.

Usage:
    python pi_audio_test.py http://<pi-ip>:5001/audio.pcm
"""
import sys
import time

from audio_cues import NoiseMonitor

BAR_WIDTH = 40


def main():
    if len(sys.argv) != 2:
        print("Usage: python pi_audio_test.py http://<pi-ip>:5001/audio.pcm")
        sys.exit(1)

    url = sys.argv[1]
    print(f"Connecting to {url} ... Ctrl+C to stop.\n")

    monitor = NoiseMonitor(url)
    monitor.start()
    try:
        while True:
            time.sleep(1.0)
            level = monitor.current_level()
            dist = monitor.distribution()
            bar = {"quiet": "-", "moderate": "#", "loud": "!"}[level] * 10
            print(f"\rlevel={level:9s} [{bar:<10s}]  session so far: {dist}   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
