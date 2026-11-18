"""
Phase 4: noise-level cue from microphone audio.

Unlike face/body cues, a single room mic can't be attributed to one
student -- noise level is inherently a CLASS-level signal, not per-student.
It's also more ambiguous than face/body cues: silence can mean either
focused, quiet work OR total disengagement, and moderate chatter can mean
either an active discussion OR off-task talking. Because that relationship
isn't well-defined without real calibration data, this module deliberately
does NOT blend noise into the per-student engagement score (unlike
body_cues.py's combine_scores()). It's recorded as its own class-level
signal for the teacher to interpret, same spirit as the "Noise Level" cue
being its own category on the slides, not a sub-component of another cue.

MVP: simple RMS/dBFS energy-level classification, not full speech-specific
VAD (voice activity detection) -- the cue we care about is overall ambient
loudness, not whether it's specifically speech. Thresholds are rough
starting points, same situation as every other cue in this project until
tuned against real classroom recordings.

NoiseMonitor pulls audio from the PI's mic over the network
(pi_stream_server.py's /audio.pcm route), not the laptop's own mic -- same
"Pi captures, laptop computes" split as video. noise_test.py still uses
the laptop's local mic directly for quick standalone sanity checks; this
class is what the real classroom_session.py uses.
"""
import threading
import time

import httpx
import numpy as np

QUIET_DBFS    = -40.0   # below this: near-silence
LOUD_DBFS     = -20.0   # above this: loud / likely disruptive

_LEVEL_SCORE = {"quiet": 0.3, "moderate": 1.0, "loud": 0.2}


def rms_dbfs(pcm_int16):
    """RMS level of an int16 PCM audio chunk, in dBFS (0 dBFS = full scale).
    Silence (all zeros) returns a very low floor instead of -inf."""
    if len(pcm_int16) == 0:
        return -96.0
    rms = np.sqrt(np.mean(pcm_int16.astype(np.float64) ** 2))
    if rms < 1e-6:
        return -96.0
    return float(20 * np.log10(rms / 32768.0))


def classify_noise(dbfs):
    """Returns (level_label, score). Score is a rough, UNCALIBRATED proxy
    for 'is this noise level consistent with active engagement' -- moderate
    (discussion-level) noise scores highest, both silence and loud/disruptive
    noise score lower. Not blended into student scores -- see module docstring."""
    if dbfs < QUIET_DBFS:
        level = "quiet"
    elif dbfs > LOUD_DBFS:
        level = "loud"
    else:
        level = "moderate"
    return level, _LEVEL_SCORE[level]


class NoiseMonitor:
    """Reads the Pi's raw PCM audio stream (pi_stream_server.py's
    /audio.pcm) on a background thread and classifies it in ~0.5s windows,
    so it never blocks the video loop. Accumulates a session-long
    quiet/moderate/loud distribution, same shape as the per-student
    pose/expression/posture distributions.

    `stream_url=None` disables audio monitoring entirely (e.g. when
    CAMERA_SOURCE is a plain laptop-webcam index during dev/testing and
    there's no Pi to stream from) rather than silently falling back to the
    laptop's own mic, which would defeat the point of this being the Pi's mic."""

    SAMPLE_RATE = 16000
    ANALYSIS_SECONDS = 0.5
    _ANALYSIS_BYTES = int(SAMPLE_RATE * ANALYSIS_SECONDS * 2)   # int16 = 2 bytes/sample

    def __init__(self, stream_url=None):
        self._stream_url = stream_url
        self._counts = {"quiet": 0, "moderate": 0, "loud": 0}
        self._last_level = "quiet"
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        if self._stream_url is None:
            print("  [INFO] No Pi audio stream configured -- noise monitoring disabled for this session.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            try:
                with httpx.stream("GET", self._stream_url, timeout=None) as response:
                    buffer = b""
                    for chunk in response.iter_bytes():
                        if not self._running:
                            return
                        buffer += chunk
                        while len(buffer) >= self._ANALYSIS_BYTES:
                            block, buffer = buffer[:self._ANALYSIS_BYTES], buffer[self._ANALYSIS_BYTES:]
                            pcm = np.frombuffer(block, dtype=np.int16)
                            level, _ = classify_noise(rms_dbfs(pcm))
                            with self._lock:
                                self._counts[level] += 1
                                self._last_level = level
            except Exception as exc:
                if self._running:
                    print(f"  [WARN] Pi audio stream unreachable ({exc}) -- retrying in 3s...")
                    time.sleep(3)

    def stop(self):
        self._running = False

    def current_level(self):
        with self._lock:
            return self._last_level

    def distribution(self):
        with self._lock:
            total = sum(self._counts.values()) or 1
            return {k: round(v / total * 100) for k, v in self._counts.items()}
