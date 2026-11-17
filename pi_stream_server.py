"""
Runs ON THE RASPBERRY PI (copy this file + kiosk.html onto the Pi and run
it there -- not on the laptop). Combines everything the Pi needs to serve
into ONE process on ONE port, so there's a single systemd service to
auto-start on boot (see pi_setup/):

    GET  /stream.mjpg    -- MJPEG video from the CSI camera (Picamera2)
    GET  /audio.pcm      -- raw PCM audio from the Pi's mic (sounddevice)
    GET  /                the touchscreen's local kiosk page (kiosk.html)
    GET  /kiosk/state     the kiosk page's own poll endpoint (JSON)
    POST /kiosk           the laptop pushes session state + color here

All compute (MediaPipe, YOLO, ArcFace, scoring) stays on the laptop -- the
Pi only captures/streams and displays. This is the same "Pi stays a thin
sensor + actuator, laptop is the single source of truth" split as every
other piece of this project.

No codec/container for audio or a fancy protocol for the kiosk state --
raw PCM and plain polling, same "simplest possible" philosophy as the
MJPEG video stream. A WebSocket would be smoother but is one more thing
that can silently disconnect right before a demo; 1x/second polling is
plenty responsive for a color change and far more robust.

One-time setup on the Pi:
    sudo apt install portaudio19-dev
    /usr/bin/python3 -m pip install flask picamera2 sounddevice numpy

Run (or let the systemd service run it):
    /usr/bin/python3 pi_stream_server.py
"""
import io
import queue
import threading

import numpy as np
import sounddevice as sd
from flask import Flask, Response, jsonify, request, send_from_directory
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

PORT = 5001
AUDIO_SAMPLE_RATE = 16000
AUDIO_BLOCK_FRAMES = 1600   # 0.1s chunks at 16kHz

app = Flask(__name__, static_folder=None)

# ─────────────────────────────────────────────────────────────────────────
# Video (MJPEG) -- the standard Picamera2 streaming pattern: a background
# recording continuously writes JPEG frames into a shared buffer, and each
# HTTP client's generator just waits on the condition variable for the next
# one. Handles multiple simultaneous viewers correctly.
# ─────────────────────────────────────────────────────────────────────────

class _StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (1280, 720)}))
_video_output = _StreamingOutput()
picam2.start_recording(MJPEGEncoder(), FileOutput(_video_output))


def _mjpeg_frames():
    while True:
        with _video_output.condition:
            _video_output.condition.wait()
            frame = _video_output.frame
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


@app.route("/stream.mjpg")
def stream_mjpg():
    return Response(_mjpeg_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ─────────────────────────────────────────────────────────────────────────
# Audio (raw PCM) -- continuous int16 mono capture from the Pi's mic,
# pushed into a bounded queue by a callback-driven InputStream, drained by
# whichever HTTP client is currently reading /audio.pcm. If a client falls
# behind, we drop audio rather than let the queue (and latency) grow
# unbounded -- a live engagement signal that's occasionally missing a
# fraction of a second is fine; one that's minutes behind is useless.
# ─────────────────────────────────────────────────────────────────────────

_audio_queue = queue.Queue(maxsize=50)


def _audio_callback(indata, frames, time_info, status):
    try:
        _audio_queue.put_nowait(bytes(indata))
    except queue.Full:
        pass


_audio_stream = sd.InputStream(
    samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype="int16",
    blocksize=AUDIO_BLOCK_FRAMES, callback=_audio_callback,
)
_audio_stream.start()


def _pcm_chunks():
    while True:
        yield _audio_queue.get()


@app.route("/audio.pcm")
def audio_pcm():
    return Response(_pcm_chunks(), mimetype=f"audio/L16; rate={AUDIO_SAMPLE_RATE}; channels=1")


# ─────────────────────────────────────────────────────────────────────────
# Kiosk display (touchscreen) -- idle/welcome, or a live full-screen color
# matching current class engagement (temporary stand-in for the LED strip
# until that hardware is bought -- see led_controller.py on the laptop side).
# ─────────────────────────────────────────────────────────────────────────

_kiosk_state = {"state": "idle", "level": "", "r": 0.0, "g": 0.0, "b": 0.0}
_kiosk_lock = threading.Lock()


@app.route("/")
def kiosk_page():
    return send_from_directory(".", "kiosk.html")


@app.route("/kiosk/state")
def kiosk_state():
    with _kiosk_lock:
        return jsonify(_kiosk_state)


@app.route("/kiosk", methods=["POST"])
def kiosk_update():
    data = request.get_json(force=True)
    with _kiosk_lock:
        if "state" in data:
            _kiosk_state["state"] = data["state"]
        if "level" in data:
            _kiosk_state["level"] = data["level"]
        if "r" in data:
            _kiosk_state["r"] = float(data["r"])
            _kiosk_state["g"] = float(data["g"])
            _kiosk_state["b"] = float(data["b"])
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
