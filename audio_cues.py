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
"""
import threading

import numpy as np
import sounddevice as sd

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
    """Runs mic capture + classification on a background audio callback
    thread (via sounddevice/PortAudio) so it never blocks the video loop.
    Accumulates a session-long quiet/moderate/loud distribution, same shape
    as the per-student pose/expression/posture distributions."""

    def __init__(self, sample_rate=16000, chunk_seconds=0.5, device=None):
        self._counts = {"quiet": 0, "moderate": 0, "loud": 0}
        self._last_level = "quiet"
        self._lock = threading.Lock()
        self._stream = sd.InputStream(
            samplerate=sample_rate, channels=1, dtype="int16",
            blocksize=int(sample_rate * chunk_seconds),
            device=device, callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        level, _ = classify_noise(rms_dbfs(indata[:, 0]))
        with self._lock:
            self._counts[level] += 1
            self._last_level = level

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()

    def current_level(self):
        with self._lock:
            return self._last_level

    def distribution(self):
        with self._lock:
            total = sum(self._counts.values()) or 1
            return {k: round(v / total * 100) for k, v in self._counts.items()}
