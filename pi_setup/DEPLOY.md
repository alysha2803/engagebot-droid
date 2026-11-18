# Deploying to the Raspberry Pi — step by step

Do these in order. Each step says how to check it actually worked before
moving to the next one — don't skip the checks, since a problem is much
easier to find right where it happened than after five more steps.

All commands below are run **on the Pi**, over SSH, unless marked otherwise.

---

## 0. Which files go where

From your laptop's `engagebot-droid` folder, these four things need to end
up on the Pi, inside a folder called `engagebot-droid` in your Pi user's
home directory (so the full path is `/home/pi/engagebot-droid/`, assuming
your Pi username is `pi` — if it's something else, use that instead
everywhere `pi` appears below):

- `pi_stream_server.py`
- `kiosk.html`
- `pi_setup/droid-stream.service`
- `pi_setup/droid-kiosk.desktop`

Nothing else from the laptop's `engagebot-droid` folder needs to go on the
Pi — everything else (`classroom_session.py`, `enroll_student.py`, the
ONNX model, etc.) runs on the laptop only.

**If you already have an `engagebot-droid` folder on the Pi from the
camera setup earlier in this project**, reuse it — just add/replace these
four files in it, you don't need a fresh folder.

### Getting the files across

From a PowerShell window **on your laptop** (not the Pi), replace
`<pi-ip>` with your Pi's actual IP address:

```powershell
cd C:\Users\Alysha\Downloads\engagebot-droid
scp pi_stream_server.py kiosk.html pi@192.168.100.41:/home/pi/engagebot-droid/
scp pi_setup/droid-stream.service pi_setup/droid-kiosk.desktop pi@192.168.100.41:/home/pi/engagebot-droid/
```

It'll ask for your Pi's password each time. If `scp` isn't recognized,
Windows' built-in OpenSSH client isn't installed — in that case, just open
each file in VS Code/Notepad, copy its contents, and paste them into a new
file on the Pi via `nano <filename>` over SSH (slower, but works
identically — this is apparently how `pi_stream_server.py` was first
created on the Pi anyway).

---

## 1. Install dependencies (one-time)

SSH into the Pi, then:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-flask portaudio19-dev
/usr/bin/python3 -m pip install sounddevice numpy --break-system-packages
```

Notes:
- Use `/usr/bin/python3` specifically, not plain `python3` — same reason as
  before in this project: `python3` may resolve to a pyenv shim, and
  `picamera2` only works with the system Python apt installs into.
- `python3-picamera2` and `python3-flask` come via `apt`, not `pip` —
  `picamera2` specifically needs the system libcamera bindings that only
  the apt package provides; a pip-installed copy generally won't work.
- `--break-system-packages` is needed on Raspberry Pi OS Bookworm because
  it blocks plain `pip install` outside a virtual environment by default.
  This is a deliberate, safe override here, not a workaround for an error.

**Check it worked:**
```bash
/usr/bin/python3 -c "import flask, picamera2, sounddevice, numpy; print('all imports OK')"
```
If that prints `all imports OK` with no errors, you're good.

---

## 2. Find your mic+speaker device's name

```bash
arecord -l
```
This lists your sound capture devices — note the **card number** for your
mic+speaker module (not the Pi's HDMI audio output, which may also show up
here). You'll likely see something like `card 1: ...`.

Also check what `sounddevice` itself sees:
```bash
/usr/bin/python3 -c "import sounddevice as sd; print(sd.query_devices())"
```
The line with a `>` in front of it is the current *default* input device.
If that's already your mic+speaker module, you don't need to change
anything. If it's the wrong device, come back to me with what this
command printed and I'll add a config option to pin the right one — I
deliberately didn't add that yet since I don't know your exact hardware's
device name from here.

---

## 3. Run it manually first (before setting up auto-start)

This step is important: running it by hand first means you see any error
directly in the terminal, instead of it failing silently inside a
background service later.

```bash
cd /home/pi/engagebot-droid
/usr/bin/python3 pi_stream_server.py
```

You should see it print `Starting camera...`-type output with no
tracebacks, and it should just sit there running (that's correct — it's a
server, it doesn't exit).

**Check it worked**, from your **laptop's web browser** (not the Pi),
replacing `<pi-ip>`:

- `http://192.168.100.41:5001/stream.mjpg` → should show the live camera feed
  (you've already proven this part works earlier in the project).
- `http://192.168.100.41:5001/` → should show the **idle/welcome kiosk screen**
  (dark green background, "EngageBot / Ready — waiting for class to
  start"). If you see this in your laptop's browser, the kiosk page
  itself is working correctly — independent of whether the Pi's own
  touchscreen can show it yet.
- `http://192.168.100.41:5001/audio.pcm` → the browser will likely just start
  downloading raw gibberish-looking data or show a blank page — that's
  actually correct, it means the audio stream is alive (a browser can't
  play raw unlabeled PCM directly). To actually verify sound quality, open
  a **second** terminal on your **laptop** instead and run:
  ```powershell
  cd C:\Users\Alysha\Downloads\engagebot-droid
  .venv_check\Scripts\python.exe pi_audio_test.py http://192.168.100.41:5001/audio.pcm
  ```
  Talk/clap near the Pi's mic — the printed level should visibly change
  between `quiet`, `moderate`, and `loud`.

Once all three check out, go back to the Pi's terminal and press
`Ctrl+C` to stop the manual run.

---

## 4. Set up auto-start on boot

### 4a. Make the Pi boot straight to desktop, no login needed

```bash
sudo raspi-config
```
Navigate: **System Options → Boot / Auto Login → Desktop Autologin** →
Finish → reboot when it asks.

### 4b. Install the background server as a systemd service

```bash
sudo cp /home/pi/engagebot-droid/pi_setup/droid-stream.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable droid-stream.service
sudo systemctl start droid-stream.service
```

**Check it worked:**
```bash
sudo systemctl status droid-stream.service
```
Should say `active (running)` in green. If your Pi username isn't `pi`,
or you put the folder somewhere else, edit
`/etc/systemd/system/droid-stream.service` first (`User=` and
`WorkingDirectory=` lines) before running the commands above.

### 4c. Install the kiosk autostart entry

```bash
mkdir -p ~/.config/autostart
cp /home/pi/engagebot-droid/pi_setup/droid-kiosk.desktop ~/.config/autostart/
```

No "check it worked" step here on its own — you'll confirm it in step 5
below, since it only does anything on a fresh boot into the desktop.

---

## 5. The real test: power-cycle it

This is the actual test of "turn it on and it just works" — don't just
run `sudo reboot`, physically **unplug and replug the Pi's power**, since
that's exactly what will happen on demo day.

**Check it worked:**
- The Pi's own touchscreen should, after boot, show the idle/welcome
  screen **on its own**, with nobody touching a keyboard.
- From your laptop, `http://192.168.100.41:5001/stream.mjpg` should work without
  you having SSH'd in or run anything manually.

Do this power-cycle test **at least twice**, ideally a few days before
your actual demo, not the morning of — boot timing issues (the touchscreen
trying to load the page before the server's fully up) are exactly the
kind of thing that works on the 1st try and fails on the 2nd.

---

## 6. Point the laptop at it

Nothing new here if `CAMERA_SOURCE` in `engagebot-droid/.env` on your
laptop already points at `http://<pi-ip>:5001/stream.mjpg` — the laptop
automatically derives the audio and kiosk-display URLs from that same
address now, since all three live on the same Pi server.

Run `classroom_session.py` as usual. When it starts, the Pi's touchscreen
should switch from idle to "Session started… Identifying students," then
to a live color once a student's recognized. When you press `Q` to stop,
it should switch back to idle.

---

## Troubleshooting

### A leftover sound/voice plays on boot (from a previous project)

Something else was set to autostart before this project and is still
configured to run. Work through these in order — the first one is the
most likely culprit, since it's the exact same mechanism this project
just added:

```bash
# 1. Other autostart entries for your user (most likely spot)
ls ~/.config/autostart/
cat ~/.config/autostart/<suspicious-file>.desktop   # inspect before deleting
rm ~/.config/autostart/<suspicious-file>.desktop

# 2. System-wide autostart entries (less likely, but check)
ls /etc/xdg/autostart/

# 3. Services enabled at boot
systemctl list-unit-files --state=enabled
systemctl cat <suspicious-service-name>             # inspect before disabling
sudo systemctl disable <suspicious-service-name>
sudo systemctl stop <suspicious-service-name>

# 4. User-level services
systemctl --user list-unit-files --state=enabled

# 5. A cron job that runs at boot
crontab -l

# 6. Old-style boot script
cat /etc/rc.local

# 7. Shell startup files (less likely for a boot-time voice clip, but
#    worth a quick check if nothing above explains it)
grep -iE "play|aplay|mpg123|omxplayer|mpv" ~/.bashrc ~/.profile 2>/dev/null
```

A voice clip like "trying to connect to internet" is almost always played
via `aplay`, `mpg123`, `omxplayer`, or `mpv` from a script — searching for
those command names in whatever you find at step 1 or 3 should point
straight at it.

### The kiosk display takes longer to appear than the camera

Expected, and partly unavoidable: the camera/audio server starts as a
system service, which can begin before the desktop session is even fully
up, while the kiosk display needs the *entire* desktop (autologin, X/
Wayland session, then its own autostart entry) to finish loading first —
a slower path by nature. Two things narrow the gap:

1. **`droid-kiosk.desktop` has been updated** (re-copy it to the Pi) to
   actively poll for the server instead of blindly waiting a fixed 8
   seconds, and to pass Chromium some startup-speed flags (skip first-run
   setup, disable background network/update checks, RAM-backed cache) —
   Chromium's own startup overhead was likely the bigger part of the delay,
   not the wait itself.
2. Re-install the autostart entry and re-test:
   ```bash
   cp /home/pi/engagebot-droid/pi_setup/droid-kiosk.desktop ~/.config/autostart/
   ```
   Then power-cycle again (not `sudo reboot` — see step 5 above for why).

If it's still noticeably slower than the camera after this, that's the
realistic floor for "boot to desktop, then launch a browser" on this
hardware — the remaining gap isn't a bug to chase further, just how long
that path takes.
