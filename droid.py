"""
EngageBot Droid Prototype
Laptop camera → MediaPipe Face Mesh → engagement score → MongoDB via API

Usage:
    python droid.py

Install dependencies first:
    pip install -r requirements.txt
"""

import os
import sys
import time
import threading
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import httpx
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Config — edit API_BASE if your server runs on a different port.
# DEFAULT_EMAIL/DEFAULT_PASSWORD/CAMERA_SOURCE come from .env (see .env.example)
# so no credential lives in this file / git history.
# ─────────────────────────────────────────────────────────────────────────────

API_BASE           = os.getenv("API_BASE", "http://localhost:5000/api")
POST_INTERVAL      = 30   # seconds between live API updates
DROID_ID           = "laptop-prototype"
TIMELINE_INTERVAL  = 5    # seconds between timeline sample points

# Video source: 0 = laptop's own webcam. To use the Raspberry Pi camera instead,
# point this at the Pi's MJPEG stream server (see pi_stream_server.py on the Pi).
CAMERA_SOURCE      = os.getenv("CAMERA_SOURCE", "0")

# Pre-fill for quick testing — leave unset in .env to be prompted instead
DEFAULT_EMAIL    = os.getenv("DROID_EMAIL", "")
DEFAULT_PASSWORD = os.getenv("DROID_PASSWORD", "")

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe landmark indices
# ─────────────────────────────────────────────────────────────────────────────

# 6-point face model for solvePnP head pose (matched to generic 3-D geometry)
_HEAD_LAND_IDS = [4, 152, 263, 33, 287, 57]
_HEAD_3D = np.array([
    [  0.0,    0.0,    0.0],   # 4   nose tip
    [  0.0, -330.0,  -65.0],   # 152 chin
    [-225.0,  170.0, -135.0],  # 263 left eye outer corner
    [ 225.0,  170.0, -135.0],  # 33  right eye outer corner
    [-150.0, -150.0, -125.0],  # 287 left mouth corner
    [ 150.0, -150.0, -125.0],  # 57  right mouth corner
], dtype=np.float64)

# Eye aspect ratio — 6 landmarks per eye
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Mouth aspect ratio — inner lip + corners
_MOUTH_TOP   = 13
_MOUTH_BOT   = 14
_MOUTH_LEFT  = 61
_MOUTH_RIGHT = 291

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds  (tune against your own face if needed)
# ─────────────────────────────────────────────────────────────────────────────

EAR_DROWSY  = 0.22   # eyes half-closed
MAR_YAWN    = 0.50   # mouth wide open

PITCH_DOWN  = -12    # degrees — looking at desk / phone
YAW_SIDE    =  20    # degrees — slightly turned
YAW_AWAY    =  35    # degrees — clearly looking away

# ─────────────────────────────────────────────────────────────────────────────
# Scoring matrix  [head_pose][expression_col] → 0.0–1.0
#
# Head pose rows :  forward | down | side | away
# Expression cols:  attentive (0) | neutral (1) | drowsy (2) | yawning (3)
# ─────────────────────────────────────────────────────────────────────────────

_MATRIX = {
    "forward": [1.00, 0.80, 0.45, 0.15],
    "down":    [0.55, 0.40, 0.25, 0.10],
    "side":    [0.25, 0.20, 0.15, 0.05],
    "away":    [0.05, 0.05, 0.05, 0.00],
}
_EXPR_COL = {"attentive": 0, "neutral": 1, "drowsy": 2, "yawning": 3}

# ─────────────────────────────────────────────────────────────────────────────
# Display colours (BGR)
# ─────────────────────────────────────────────────────────────────────────────

_COL_GREEN  = (50,  205, 120)
_COL_ORANGE = (0,   165, 255)
_COL_RED    = (60,   80, 220)
_COL_WHITE  = (255, 255, 255)
_COL_DARK   = (30,   30,  30)

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pt(lm, idx, w, h):
    return np.array([lm[idx].x * w, lm[idx].y * h])


def _ear(lm, ids, w, h):
    """Eye Aspect Ratio from 6 eye landmarks."""
    p = [_pt(lm, i, w, h) for i in ids]
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    return (A + B) / (2.0 * C + 1e-6)


def _mar(lm, w, h):
    """Mouth Aspect Ratio — vertical opening / horizontal width."""
    top   = _pt(lm, _MOUTH_TOP,   w, h)
    bot   = _pt(lm, _MOUTH_BOT,   w, h)
    left  = _pt(lm, _MOUTH_LEFT,  w, h)
    right = _pt(lm, _MOUTH_RIGHT, w, h)
    return np.linalg.norm(top - bot) / (np.linalg.norm(left - right) + 1e-6)


def _head_pose(lm, w, h):
    """Returns (pitch, yaw) in degrees using solvePnP."""
    pts2d = np.array(
        [[lm[i].x * w, lm[i].y * h] for i in _HEAD_LAND_IDS],
        dtype=np.float64,
    )
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        _HEAD_3D, pts2d, cam, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1]   # pitch, yaw


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify(pitch, yaw, ear_avg, mar_val):
    """
    Returns (head_pose_label, expression_label, score 0–1).

    Head pose:
        forward — looking at screen
        down    — looking at desk / phone
        side    — slightly turned
        away    — clearly looking elsewhere

    Expression:
        attentive — eyes wide open
        neutral   — normal resting
        drowsy    — eyes heavy
        yawning   — mouth wide open
    """
    # Head pose
    if abs(yaw) >= YAW_AWAY:
        pose = "away"
    elif abs(yaw) >= YAW_SIDE:
        pose = "side"
    elif pitch <= PITCH_DOWN:
        pose = "down"
    else:
        pose = "forward"

    # Expression
    if mar_val >= MAR_YAWN:
        expr = "yawning"
    elif ear_avg <= EAR_DROWSY:
        expr = "drowsy"
    elif ear_avg >= 0.30:
        expr = "attentive"
    else:
        expr = "neutral"

    score = _MATRIX[pose][_EXPR_COL[expr]]
    return pose, expr, score


def score_to_level(score):
    if score >= 0.65:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# Overlay drawing
# ─────────────────────────────────────────────────────────────────────────────

def _level_colour(level):
    return {
        "high":   _COL_GREEN,
        "medium": _COL_ORANGE,
        "low":    _COL_RED,
    }.get(level, _COL_WHITE)


def draw_overlay(frame, pose, expr, score, level, avg30, no_face=False):
    h, w = frame.shape[:2]

    # Dark translucent header bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), _COL_DARK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    if no_face:
        cv2.putText(frame, "No face detected — make sure your face is visible",
                    (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COL_RED, 2)
        return

    col = _level_colour(level)

    # Line 1 — engagement level + instant score
    cv2.putText(
        frame,
        f"Engagement: {level.upper()}   score={int(score * 100)}%   30s avg={int(avg30 * 100)}%",
        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, col, 2,
    )
    # Line 2 — raw signals
    cv2.putText(
        frame,
        f"pose={pose}   expr={expr}   EAR={ear_avg_display:.2f}   MAR={mar_display:.2f}",
        (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COL_WHITE, 1,
    )

    # Score progress bar
    bar_x1, bar_x2 = 12, w - 12
    bar_y1, bar_y2 = 66, 74
    filled = int((bar_x2 - bar_x1) * score)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (70, 70, 70), -1)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + filled, bar_y2), col, -1)

    # Threshold markers on bar
    for thresh, label in [(0.35, ""), (0.65, "")]:
        mx = bar_x1 + int((bar_x2 - bar_x1) * thresh)
        cv2.line(frame, (mx, bar_y1 - 2), (mx, bar_y2 + 2), _COL_WHITE, 1)


# Module-level mutable for overlay (avoids passing through classify path)
ear_avg_display = 0.0
mar_display     = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# API client
# ─────────────────────────────────────────────────────────────────────────────

class EngageBotAPI:
    def __init__(self):
        self._client = httpx.Client(base_url=API_BASE, timeout=10)

    def login(self, email, password):
        r = self._client.post(
            "/auth/teacher/login",
            json={"email": email, "password": password},
        )
        r.raise_for_status()
        token = r.json()["token"]
        self._client.headers.update({"Authorization": f"Bearer {token}"})

    def profile(self):
        r = self._client.get("/teachers/profile")
        r.raise_for_status()
        return r.json()

    def schedules(self, teacher_id):
        r = self._client.get("/schedules", params={"teacherId": teacher_id})
        r.raise_for_status()
        return r.json()

    def students(self, class_group):
        r = self._client.get("/students", params={"classGroup": class_group})
        r.raise_for_status()
        return r.json()

    def create_report(self, payload):
        r = self._client.post("/reports", json=payload)
        r.raise_for_status()
        return r.json()

    def update_report(self, report_id, patch):
        r = self._client.patch(f"/reports/{report_id}", json=patch)
        r.raise_for_status()
        return r.json()

    def close(self):
        self._client.close()


# ─────────────────────────────────────────────────────────────────────────────
# Session — manages one in-progress report and its periodic flush
# ─────────────────────────────────────────────────────────────────────────────

class Session:
    def __init__(self, api, schedule, student, teacher_name):
        self._api               = api
        self._schedule          = schedule
        self._student           = student
        self._teacher_name      = teacher_name
        self._report_id         = None
        self._buf               = []    # samples since last flush
        self._all_scores        = []    # every sample for the whole session
        self._all_distracted    = 0     # cumulative distracted-frame count
        self._pose_counts       = {}    # {pose: frame_count}
        self._expr_counts       = {}    # {expr: frame_count}
        self._ear_samples       = []
        self._mar_samples       = []
        self._timeline          = []    # [{t: secs, score: 0-100}]
        self._last_tl_t         = -TIMELINE_INTERVAL
        self._start_ts          = None
        self._lock              = threading.Lock()
        self._running           = False

    def start(self):
        self._start_ts = time.time()
        now = datetime.now()
        resp = self._api.create_report({
            "date":              now.strftime("%Y-%m-%d"),
            "subject":           self._schedule["subject"],
            "classGroup":        self._schedule["classGroup"],
            "teacherName":       self._teacher_name,
            "droidId":           DROID_ID,
            "startTime":         now.strftime("%H:%M"),
            "status":            "in_progress",
            "avgFocusScore":     0,
            "overallEngagement": "medium",
            "studentEngagements": [],
        })
        self._report_id = resp["id"]
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        print(f"  Report created (id: {self._report_id})")
        print(f"  Pushing updates every {POST_INTERVAL}s — press Q in the camera window to stop.\n")

    def record(self, score, pose="", expr="", ear=0.0, mar=0.0):
        elapsed = int(time.time() - self._start_ts) if self._start_ts else 0
        with self._lock:
            self._buf.append(score)
            self._all_scores.append(score)
            if score < 0.35:
                self._all_distracted += 1
            if pose:
                self._pose_counts[pose] = self._pose_counts.get(pose, 0) + 1
            if expr:
                self._expr_counts[expr] = self._expr_counts.get(expr, 0) + 1
            if ear > 0:
                self._ear_samples.append(ear)
            if mar > 0:
                self._mar_samples.append(mar)
            if elapsed >= self._last_tl_t + TIMELINE_INTERVAL:
                self._timeline.append({"t": elapsed, "score": int(score * 100)})
                self._last_tl_t = elapsed

    def rolling_avg(self):
        with self._lock:
            if self._all_scores:
                return sum(self._all_scores) / len(self._all_scores)
            return 0.0

    def stop(self):
        self._running = False
        self._flush(final=True)

    # ── internal ──────────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            time.sleep(POST_INTERVAL)
            if self._running:
                self._flush()

    def _flush(self, final=False):
        with self._lock:
            buf = list(self._buf)
            self._buf.clear()

        # Accumulate into session totals.
        if buf:
            self._all_scores.extend(buf)
            self._all_distracted += sum(1 for s in buf if s < 0.35)

        # Skip mid-session flushes with nothing new.
        if not buf and not final:
            return

        patch = {}

        # Build student engagement from cumulative session data.
        if self._all_scores:
            session_avg  = sum(self._all_scores) / len(self._all_scores)
            focus_int    = int(session_avg * 100)
            level        = score_to_level(session_avg)
            student_id   = self._student.get("id") or self._student.get("_id", "")
            student_name = self._student.get("name", "Unknown")

            total_pose = sum(self._pose_counts.values()) or 1
            total_expr = sum(self._expr_counts.values()) or 1

            patch = {
                "avgFocusScore":      focus_int,
                "overallEngagement":  level,
                "studentEngagements": [{
                    "studentId":             student_id,
                    "studentName":           student_name,
                    "engagementLevel":       level,
                    "focusScore":            focus_int,
                    "distractedCount":       self._all_distracted,
                    "participationScore":    focus_int,
                    "avgEAR":                round(sum(self._ear_samples) / len(self._ear_samples), 3) if self._ear_samples else 0,
                    "avgMAR":                round(sum(self._mar_samples) / len(self._mar_samples), 3) if self._mar_samples else 0,
                    "poseDistribution":      {k: round(v / total_pose * 100) for k, v in self._pose_counts.items()},
                    "expressionDistribution":{k: round(v / total_expr * 100) for k, v in self._expr_counts.items()},
                    "timeline":              list(self._timeline),
                }],
            }

        # Always mark completed on final, even with no camera data.
        if final:
            patch["status"]  = "completed"
            patch["endTime"] = datetime.now().strftime("%H:%M")

        if not patch:
            return

        try:
            self._api.update_report(self._report_id, patch)
            tag = "FINAL" if final else "UPDATE"
            if self._all_scores:
                focus_int = int(sum(self._all_scores) / len(self._all_scores) * 100)
                level = score_to_level(sum(self._all_scores) / len(self._all_scores))
                student_name = self._student.get("name", "Unknown")
                print(
                    f"  [{tag}]  score={focus_int}%  level={level}"
                    f"  total_samples={len(self._all_scores)}  student={student_name}"
                )
            else:
                print(f"  [{tag}]  no samples — session marked completed")
        except Exception as exc:
            print(f"  [WARN] API update failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI setup — login, pick schedule, pick student
# ─────────────────────────────────────────────────────────────────────────────

def _ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else str(default) if default is not None else ""


def setup(api):
    print("\n══════════════════════════════════════════")
    print("   EngageBot Droid Prototype")
    print("══════════════════════════════════════════\n")

    email    = _ask("Teacher email",    DEFAULT_EMAIL    or None)
    password = _ask("Teacher password", DEFAULT_PASSWORD or None)

    print("\n  Logging in … ", end="", flush=True)
    try:
        api.login(email, password)
    except httpx.HTTPStatusError as e:
        print(f"FAILED ({e.response.status_code})")
        print(f"  {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"FAILED\n  {e}")
        sys.exit(1)
    print("OK")

    profile      = api.profile()
    teacher_id   = profile["id"]
    teacher_name = profile["name"]
    print(f"  Signed in as: {teacher_name}\n")

    # Pick schedule
    schedules = api.schedules(teacher_id)
    if not schedules:
        print("  No schedules found. Ask an admin to assign classes first.")
        sys.exit(1)

    print("  Your schedules:")
    for i, s in enumerate(schedules, 1):
        print(
            f"    [{i}]  {s['classGroup']}  ·  {s['subject']}"
            f"  ·  {s.get('day', '')}  {s.get('startTime', '')}–{s.get('endTime', '')}"
        )

    idx = int(_ask("\n  Select schedule number", 1)) - 1
    if idx < 0 or idx >= len(schedules):
        print("  Invalid selection.")
        sys.exit(1)
    schedule = schedules[idx]

    # Pick student (who is sitting in front of the camera for the test)
    students = api.students(schedule["classGroup"])
    if not students:
        print(f"  No students found in {schedule['classGroup']}.")
        sys.exit(1)

    print(f"\n  Students in {schedule['classGroup']}:")
    for i, s in enumerate(students, 1):
        sid = s.get("id") or s.get("_id", "")
        print(f"    [{i}]  {s.get('name', 'Unknown')}  (id: {sid})")

    sidx = int(_ask("\n  Which student are YOU (the one in front of the camera)", 1)) - 1
    if sidx < 0 or sidx >= len(students):
        print("  Invalid selection.")
        sys.exit(1)
    student = students[sidx]

    print(f"\n  ── Session Summary ─────────────────────────────")
    print(f"  Class:   {schedule['classGroup']}  ·  {schedule['subject']}")
    print(f"  Student: {student.get('name')}  ({student.get('id') or student.get('_id', '')})")
    print(f"  Update interval: every {POST_INTERVAL}s")
    print(f"  ────────────────────────────────────────────────\n")

    input("  Press Enter to start the session …")

    return Session(api, schedule, student, teacher_name)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global ear_avg_display, mar_display

    api     = EngageBotAPI()
    session = setup(api)
    session.start()

    camera_source = int(CAMERA_SOURCE) if CAMERA_SOURCE.isdigit() else CAMERA_SOURCE
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera source ({CAMERA_SOURCE}).")
        session.stop()
        api.close()
        sys.exit(1)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Webcam read failed — stopping.")
                break

            frame = cv2.flip(frame, 1)   # mirror so left/right feel natural
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark

                ear_l = _ear(lm, _LEFT_EYE,  w, h)
                ear_r = _ear(lm, _RIGHT_EYE, w, h)
                ear_avg = (ear_l + ear_r) / 2.0
                mar_val = _mar(lm, w, h)
                pitch, yaw = _head_pose(lm, w, h)

                pose, expr, score = classify(pitch, yaw, ear_avg, mar_val)
                level = score_to_level(score)
                session.record(score, pose, expr, ear_avg, mar_val)

                # expose to overlay (module-level vars)
                ear_avg_display = ear_avg
                mar_display     = mar_val

                draw_overlay(frame, pose, expr, score, level, session.rolling_avg())
            else:
                draw_overlay(frame, "", "", 0.0, "low", session.rolling_avg(), no_face=True)

            cv2.imshow("EngageBot Droid Prototype  [Q = stop session]", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    finally:
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\n  Stopping session …")
        session.stop()
        api.close()
        print("  Done.\n")


if __name__ == "__main__":
    main()
