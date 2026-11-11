"""
Phase 6: maps the whole-class engagement level to an LED colour and pushes
it to the Pi over HTTP.

Same "laptop computes, Pi just executes I/O" split as the camera
(pi_stream_server.py streams video, all MediaPipe stays on the laptop) and
the recommended audio design (Pi captures, laptop classifies) -- the Pi
stays a thin actuator here too. This module only decides the colour and
sends it; pi_led_server.py (meant to run ON the Pi, deployed there once the
LED hardware is wired up) is the only thing that touches GPIO.

No hardware is connected yet, so LEDClient degrades gracefully: if the Pi's
LED server isn't reachable, it logs a warning once and keeps going --
it never blocks or crashes the main detection loop over a missing actuator.
"""
import os

import httpx

LED_SERVER_URL = os.getenv("LED_SERVER_URL", "http://192.168.100.41:5002")

# RGB, 0.0-1.0 per channel (matches gpiozero.RGBLED's .color convention).
# Same colour meaning as every on-screen overlay in this project:
# green=high, orange=medium, red=low.
LEVEL_TO_COLOR = {
    "high":   (0.0, 1.0, 0.0),
    "medium": (1.0, 0.5, 0.0),
    "low":    (1.0, 0.0, 0.0),
    "absent": (0.0, 0.0, 0.0),
}


def level_to_color(level):
    return LEVEL_TO_COLOR.get(level, (0.0, 0.0, 0.0))


class LEDClient:
    def __init__(self, server_url=LED_SERVER_URL, timeout=2.0):
        self._client = httpx.Client(base_url=server_url, timeout=timeout)
        self._last_level = None
        self._warned = False

    def set_level(self, level):
        """Pushes the colour for `level` to the Pi, but only when the level
        actually changed -- no point spamming the same colour every call."""
        if level == self._last_level:
            return
        color = level_to_color(level)
        try:
            self._client.post("/led", json={"r": color[0], "g": color[1], "b": color[2]})
            self._last_level = level
            self._warned = False
        except Exception as exc:
            if not self._warned:
                print(f"  [WARN] LED server unreachable ({exc}) -- continuing without it.")
                self._warned = True

    def close(self):
        self._client.close()
