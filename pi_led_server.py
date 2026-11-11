"""
Phase 6: runs ON THE PI once the LED hardware is wired up (not run on the
laptop -- deploy this file to the Pi the same way pi_stream_server.py was).

Receives a colour from led_controller.py (laptop side) and sets a single
RGB LED via gpiozero. Assumes the simplest common setup: one RGB LED (3
legs + common ground) on 3 GPIO pins, driven via PWM for colour mixing --
adjust LED_RED_PIN/LED_GREEN_PIN/LED_BLUE_PIN below (or via env vars) to
match your actual wiring. If you have an addressable strip (WS2812/
NeoPixel) instead, this script needs reworking around rpi_ws281x --
different library, different wiring, ask before assuming this still fits.

Falls back to a console-only "simulation mode" if gpiozero can't find a
real GPIO backend (e.g. running this on a laptop to sanity-check the HTTP
side before the Pi hardware exists) -- same "build ahead of hardware"
approach used for the camera and mic.

Setup on the Pi (once wired up):
    /usr/bin/python3 -m pip install flask gpiozero
    /usr/bin/python3 pi_led_server.py

Usage from the laptop side: see led_controller.py.
"""
import os

from flask import Flask, request, jsonify

LED_RED_PIN   = int(os.getenv("LED_RED_PIN", "17"))
LED_GREEN_PIN = int(os.getenv("LED_GREEN_PIN", "27"))
LED_BLUE_PIN  = int(os.getenv("LED_BLUE_PIN", "22"))
PORT          = int(os.getenv("LED_SERVER_PORT", "5002"))

try:
    from gpiozero import RGBLED
    led = RGBLED(red=LED_RED_PIN, green=LED_GREEN_PIN, blue=LED_BLUE_PIN)
    SIMULATED = False
except Exception as exc:
    print(f"gpiozero/GPIO not available ({exc}) -- running in SIMULATION mode, colours only print to console.")
    led = None
    SIMULATED = True

app = Flask(__name__)


@app.route("/led", methods=["POST"])
def set_led():
    data = request.get_json(force=True)
    r, g, b = float(data["r"]), float(data["g"]), float(data["b"])
    if SIMULATED:
        print(f"[SIMULATED LED] color = ({r:.2f}, {g:.2f}, {b:.2f})")
    else:
        led.color = (r, g, b)
    return jsonify({"ok": True})


if __name__ == "__main__":
    print(f"LED server listening on 0.0.0.0:{PORT}  (simulated={SIMULATED})")
    app.run(host="0.0.0.0", port=PORT)
