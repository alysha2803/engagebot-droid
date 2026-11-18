"""
Phase 6 (+ touchscreen kiosk stand-in): maps the whole-class engagement
level to a colour and pushes it to the Pi over HTTP.

Same "laptop computes, Pi just executes I/O" split as the camera and audio
(pi_stream_server.py streams/displays, all MediaPipe/scoring stays on the
laptop) -- the Pi stays a thin actuator here too.

Right now this targets the touchscreen kiosk (pi_stream_server.py's
/kiosk endpoint, rendered by kiosk.html) as a TEMPORARY stand-in for a
real LED strip, which isn't bought yet. When that hardware arrives, a
real LED receiver can listen on the same /kiosk contract (or its own
endpoint) without anything on this side changing -- the color/level
decision logic stays identical either way.

LEDClient degrades gracefully if the Pi is unreachable: logs a warning
once and keeps going, never blocking or crashing the main detection loop
over a missing/offline actuator.
"""
import os

import httpx

# Defaults to the same host:port as the video/audio stream
# (pi_stream_server.py serves all three), derived automatically from
# CAMERA_SOURCE by classroom_session.py when possible -- this env var is
# just the fallback for standalone use.
KIOSK_SERVER_URL = os.getenv("KIOSK_SERVER_URL", "http://192.168.100.41:5001")

# RGB, 0.0-1.0 per channel. Same colour meaning as every on-screen overlay
# in this project: green=high, orange=medium, red=low.
LEVEL_TO_COLOR = {
    "high":   (0.0, 1.0, 0.0),
    "medium": (1.0, 0.5, 0.0),
    "low":    (1.0, 0.0, 0.0),
}


def level_to_color(level):
    return LEVEL_TO_COLOR.get(level, (0.0, 0.0, 0.0))


class LEDClient:
    """`server_url=None` disables the display push entirely (e.g. no Pi
    during laptop-webcam dev/testing) rather than pointing at a bogus URL."""

    def __init__(self, server_url=KIOSK_SERVER_URL, timeout=2.0):
        self._client = httpx.Client(base_url=server_url, timeout=timeout) if server_url else None
        self._last_level = None
        self._warned = False

    def set_level(self, level):
        """Pushes the colour + level label for `level` to the Pi, but only
        when the level actually changed -- no point spamming the same
        colour every call."""
        if level == self._last_level:
            return
        color = level_to_color(level)
        self._post({"state": "live", "level": level, "r": color[0], "g": color[1], "b": color[2]})
        self._last_level = level

    def set_idle(self):
        """Switches the Pi's display back to its idle/welcome screen --
        called when a class session ends."""
        self._post({"state": "idle"})
        self._last_level = None

    def set_starting(self):
        """Acknowledges a session has begun before any student's actually
        been recognized yet -- called once right when a session starts, so
        the display doesn't sit on 'idle' for the recognition warm-up
        period and look like nothing happened when Start was pressed."""
        self._post({"state": "live", "level": "", "r": 0.6, "g": 0.6, "b": 0.6})
        self._last_level = None

    def _post(self, payload):
        if self._client is None:
            return
        try:
            self._client.post("/kiosk", json=payload)
            self._warned = False
        except Exception as exc:
            if not self._warned:
                print(f"  [WARN] Droid display unreachable ({exc}) -- continuing without it.")
                self._warned = True

    def close(self):
        if self._client is not None:
            self._client.close()
