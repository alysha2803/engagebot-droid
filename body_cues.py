"""
Phase 3: body-language cues from YOLOv8-Pose keypoints.

Scoped to the upper body on purpose -- in a real classroom, students are
seated at desks, so hips/knees/ankles are usually occluded by the desk and
unreliable. Everything here works off nose/eyes/ears/shoulders/elbows/wrists
(COCO keypoint indices 0-10), which stay visible even seated.

Keypoint order (standard COCO-17, what YOLOv8-Pose outputs):
    0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
    5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
    9 left_wrist, 10 right_wrist,
    11 left_hip, 12 right_hip, 13 left_knee, 14 right_knee,
    15 left_ankle, 16 right_ankle

Each keypoint is (x, y, confidence). "Left"/"right" are the subject's own
left/right, same convention as droid.py's face landmarks.
"""
import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10

MIN_KP_CONF = 0.3

# Thresholds, as fractions of shoulder width (scale-invariant -- doesn't
# matter how close/far the student is from the camera).
HAND_RAISED_RATIO   = 0.5    # wrist this far above the shoulder line -> raised
HEAD_DOWN_RATIO     = 0.15   # nose sinks below shoulder midpoint -> slumped/drooping
LEAN_FORWARD_RATIO  = 0.35   # shoulder width grows vs a track's own baseline -> leaning in

_BODY_MATRIX = {
    # posture_label: score (0-1), independent signal from the face pose/expr score --
    # the caller combines both, it is not replaced by this.
    "hand_raised":   1.00,
    "leaning_in":     0.75,
    "neutral":        0.55,
    "head_down":      0.20,
    "slouched":       0.15,
}


def _kp(keypoints, idx):
    x, y, conf = keypoints[idx]
    return (x, y) if conf >= MIN_KP_CONF else None


def classify_body(keypoints, shoulder_width_baseline=None):
    """`keypoints` is a (17, 3) array from one YOLOv8-Pose detection.
    `shoulder_width_baseline` is that track's own historical average shoulder
    width (in pixels), used to detect leaning forward/back in a
    scale-invariant way; pass None on the first sighting of a track.

    Returns (posture_label, body_score, shoulder_width_px_or_None).
    """
    l_sh, r_sh = _kp(keypoints, L_SHOULDER), _kp(keypoints, R_SHOULDER)
    if l_sh is None or r_sh is None:
        return "neutral", _BODY_MATRIX["neutral"], None

    shoulder_mid = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
    shoulder_width = float(np.linalg.norm(np.array(l_sh) - np.array(r_sh)))
    if shoulder_width < 1e-3:
        return "neutral", _BODY_MATRIX["neutral"], None

    # Hand raised: either wrist well above the shoulder line.
    for wrist_idx in (L_WRIST, R_WRIST):
        wrist = _kp(keypoints, wrist_idx)
        if wrist is not None:
            raise_amount = (shoulder_mid[1] - wrist[1]) / shoulder_width
            if raise_amount >= HAND_RAISED_RATIO:
                return "hand_raised", _BODY_MATRIX["hand_raised"], shoulder_width

    # Head down: nose sinks toward/below the shoulder line (slumping over a desk).
    nose = _kp(keypoints, NOSE)
    if nose is not None:
        drop = (nose[1] - shoulder_mid[1]) / shoulder_width
        if drop >= -HEAD_DOWN_RATIO:
            return "head_down", _BODY_MATRIX["head_down"], shoulder_width

    # Leaning forward/back: compare current shoulder width to this track's
    # own baseline (bigger apparent width == closer to camera == leaning in).
    if shoulder_width_baseline and shoulder_width_baseline > 0:
        growth = (shoulder_width - shoulder_width_baseline) / shoulder_width_baseline
        if growth >= LEAN_FORWARD_RATIO:
            return "leaning_in", _BODY_MATRIX["leaning_in"], shoulder_width
        if growth <= -LEAN_FORWARD_RATIO:
            return "slouched", _BODY_MATRIX["slouched"], shoulder_width

    return "neutral", _BODY_MATRIX["neutral"], shoulder_width
