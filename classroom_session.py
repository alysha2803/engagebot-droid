"""
Phase 2: the real droid loop -- multi-face detection + tracking + recognition
+ per-student engagement scoring, writing one SessionReport covering the
WHOLE class (not a single pre-selected student like droid.py's prototype).

Pipeline per frame:
    MediaPipe (detect all faces + landmarks)
        -> IoUTracker (persist an identity across frames without
           re-running recognition every single frame)
        -> FaceRecognizer, re-checked every RECOGNITION_INTERVAL_FRAMES
           per track (best-match against the class's enrolled students)
        -> classify() from droid.py (pose/expression -> engagement score),
           reused as-is -- the per-face cue extraction doesn't change just
           because we now know whose face it is.

Each recognized student's scores are accumulated independently and flushed
to the backend periodically as studentEngagements[], same report shape
droid.py already produces (just with N students instead of 1).

Usage:
    python classroom_session.py
"""
import os
import sys
import time
from datetime import datetime

import cv2
import httpx
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from droid import (
    _LEFT_EYE, _RIGHT_EYE, _ear, _mar, _head_pose, classify, score_to_level,
    CAMERA_SOURCE as DEFAULT_CAMERA_SOURCE, EngageBotAPI,
)
from face_embedding import FaceEmbedder, five_points_from_mediapipe
from face_recognizer import FaceRecognizer
from tracker import IoUTracker

API_BASE               = os.getenv("API_BASE", "http://localhost:5000/api")
CAMERA_SOURCE           = os.getenv("CAMERA_SOURCE", DEFAULT_CAMERA_SOURCE)
DEFAULT_EMAIL           = os.getenv("DROID_EMAIL", "")
DEFAULT_PASSWORD        = os.getenv("DROID_PASSWORD", "")
MAX_FACES               = int(os.getenv("MAX_FACES", "25"))
RECOGNITION_INTERVAL    = int(os.getenv("RECOGNITION_INTERVAL_FRAMES", "10"))
POST_INTERVAL           = 30     # seconds between live API updates
TIMELINE_INTERVAL       = 5      # seconds between timeline sample points
DROID_ID                = "classroom-prototype"
DISTRACTED_CUTOFF       = 0.35

_COL_GREEN = (50, 205, 120)
_COL_ORANGE = (0, 165, 255)
_COL_RED = (60, 80, 220)
_COL_GRAY = (130, 130, 130)
_COL_WHITE = (255, 255, 255)
_COL_DARK = (30, 30, 30)
_LEVEL_COLOUR = {"high": _COL_GREEN, "medium": _COL_ORANGE, "low": _COL_RED}


def _ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else str(default) if default is not None else ""


class StudentAggregator:
    """Mirrors droid.py's Session accumulation, but one instance per student
    instead of for the single student the old prototype assumed."""

    def __init__(self, student_id, name):
        self.student_id = student_id
        self.name = name
        self.all_scores = []
        self.all_distracted = 0
        self.pose_counts = {}
        self.expr_counts = {}
        self.ear_samples = []
        self.mar_samples = []
        self.timeline = []
        self.last_tl_t = -TIMELINE_INTERVAL

    def record(self, elapsed, score, pose, expr, ear, mar):
        self.all_scores.append(score)
        if score < DISTRACTED_CUTOFF:
            self.all_distracted += 1
        self.pose_counts[pose] = self.pose_counts.get(pose, 0) + 1
        self.expr_counts[expr] = self.expr_counts.get(expr, 0) + 1
        self.ear_samples.append(ear)
        self.mar_samples.append(mar)
        if elapsed >= self.last_tl_t + TIMELINE_INTERVAL:
            self.timeline.append({"t": elapsed, "score": int(score * 100)})
            self.last_tl_t = elapsed

    def avg_score(self):
        return sum(self.all_scores) / len(self.all_scores) if self.all_scores else 0.0

    def to_engagement_dict(self):
        avg = self.avg_score()
        focus_int = int(avg * 100)
        total_pose = sum(self.pose_counts.values()) or 1
        total_expr = sum(self.expr_counts.values()) or 1
        return {
            "studentId": self.student_id,
            "studentName": self.name,
            "engagementLevel": score_to_level(avg),
            "focusScore": focus_int,
            "distractedCount": self.all_distracted,
            "participationScore": focus_int,
            "avgEAR": round(sum(self.ear_samples) / len(self.ear_samples), 3) if self.ear_samples else 0,
            "avgMAR": round(sum(self.mar_samples) / len(self.mar_samples), 3) if self.mar_samples else 0,
            "poseDistribution": {k: round(v / total_pose * 100) for k, v in self.pose_counts.items()},
            "expressionDistribution": {k: round(v / total_expr * 100) for k, v in self.expr_counts.items()},
            "timeline": list(self.timeline),
        }


def setup(api):
    print("\n══════════════════════════════════════════")
    print("   EngageBot Classroom Session (Phase 2)")
    print("══════════════════════════════════════════\n")

    email = _ask("Teacher email", DEFAULT_EMAIL or None)
    password = _ask("Teacher password", DEFAULT_PASSWORD or None)
    print("\n  Logging in … ", end="", flush=True)
    api.login(email, password)
    print("OK")

    profile = api.profile()
    teacher_id, teacher_name = profile["id"], profile["name"]
    print(f"  Signed in as: {teacher_name}\n")

    schedules = api.schedules(teacher_id)
    if not schedules:
        print("  No schedules found.")
        sys.exit(1)

    print("  Your schedules:")
    for i, s in enumerate(schedules, 1):
        print(f"    [{i}]  {s['classGroup']}  ·  {s['subject']}")
    idx = int(_ask("\n  Select schedule number", 1)) - 1
    schedule = schedules[idx]

    return teacher_name, schedule


def fetch_recognizer(api_base, headers, class_group):
    client = httpx.Client(base_url=api_base, timeout=10, headers=headers)
    r = client.get(f"/students/class/{class_group}/face-embeddings")
    r.raise_for_status()
    enrolled = r.json()
    client.close()
    for e in enrolled:
        e["embeddings"] = [np.array(v, dtype=np.float32) for v in e["embeddings"]]
    print(f"  {len(enrolled)} student(s) enrolled in {class_group}.")
    return FaceRecognizer(enrolled)


def _face_bbox(lm, w, h, pad=0.15):
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    x0, x1 = min(xs) * w, max(xs) * w
    y0, y1 = min(ys) * h, max(ys) * h
    pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
    return int(x0 - pw), int(y0 - ph), int(x1 + pw), int(y1 + ph)


def main():
    api = EngageBotAPI()
    teacher_name, schedule = setup(api)
    class_group = schedule["classGroup"]

    recognizer = fetch_recognizer(API_BASE, dict(api._client.headers), class_group)
    if recognizer.is_empty():
        print(f"  No one enrolled yet in {class_group} -- run enroll_student.py first.")
        sys.exit(1)

    embedder = FaceEmbedder()
    tracker = IoUTracker()
    aggregators = {}   # student_id -> StudentAggregator

    now = datetime.now()
    report = api.create_report({
        "date": now.strftime("%Y-%m-%d"),
        "subject": schedule["subject"],
        "classGroup": class_group,
        "teacherName": teacher_name,
        "droidId": DROID_ID,
        "startTime": now.strftime("%H:%M"),
        "status": "in_progress",
        "avgFocusScore": 0,
        "overallEngagement": "medium",
        "studentEngagements": [],
    })
    report_id = report["id"]
    start_ts = time.time()
    last_flush = start_ts
    print(f"  Report created (id: {report_id}). Press Q in the camera window to stop.\n")

    camera_source = int(CAMERA_SOURCE) if str(CAMERA_SOURCE).isdigit() else CAMERA_SOURCE
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"ERROR: could not open camera source ({camera_source}).")
        sys.exit(1)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=MAX_FACES, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    def flush(final=False):
        if not aggregators and not final:
            return
        engagements = [agg.to_engagement_dict() for agg in aggregators.values()]
        class_avg = (sum(a.avg_score() for a in aggregators.values()) / len(aggregators)) if aggregators else 0.0
        patch = {
            "avgFocusScore": int(class_avg * 100),
            "overallEngagement": score_to_level(class_avg),
            "studentEngagements": engagements,
        }
        if final:
            patch["status"] = "completed"
            patch["endTime"] = datetime.now().strftime("%H:%M")
        try:
            api.update_report(report_id, patch)
            tag = "FINAL" if final else "UPDATE"
            print(f"  [{tag}]  {len(engagements)} student(s)  class_avg={int(class_avg*100)}%")
        except Exception as exc:
            print(f"  [WARN] API update failed: {exc}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed -- stopping.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            elapsed = int(time.time() - start_ts)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            landmark_sets = result.multi_face_landmarks or []
            bboxes = [_face_bbox(lm_obj.landmark, w, h) for lm_obj in landmark_sets]
            tracks = tracker.update(bboxes)

            for track, lm_obj in zip(tracks, landmark_sets):
                lm = lm_obj.landmark

                if track.frames_since_recognition >= RECOGNITION_INTERVAL:
                    pts5 = five_points_from_mediapipe(lm, w, h)
                    emb = embedder.embed_from_frame(frame, pts5)
                    match = recognizer.recognize(emb) if emb is not None else None
                    track.student_id = match.student_id if match else None
                    track.student_name = match.name if match else None
                    track.frames_since_recognition = 0

                ear_avg = (_ear(lm, _LEFT_EYE, w, h) + _ear(lm, _RIGHT_EYE, w, h)) / 2.0
                mar_val = _mar(lm, w, h)
                pitch, yaw = _head_pose(lm, w, h)
                pose, expr, score = classify(pitch, yaw, ear_avg, mar_val)
                level = score_to_level(score)

                if track.student_id:
                    agg = aggregators.get(track.student_id)
                    if agg is None:
                        agg = StudentAggregator(track.student_id, track.student_name)
                        aggregators[track.student_id] = agg
                    agg.record(elapsed, score, pose, expr, ear_avg, mar_val)
                    col = _LEVEL_COLOUR.get(level, _COL_WHITE)
                    label = f"{track.student_name}  {int(score*100)}%"
                else:
                    col = _COL_GRAY
                    label = "Unknown"

                x0, y0, x1, y1 = track.bbox
                cv2.rectangle(frame, (x0, y0), (x1, y1), col, 2)
                cv2.rectangle(frame, (x0, y0 - 20), (x0 + 9 * len(label), y0), _COL_DARK, -1)
                cv2.putText(frame, label, (x0 + 2, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            class_avg_now = (sum(a.avg_score() for a in aggregators.values()) / len(aggregators)) if aggregators else 0.0
            header = f"Students seen: {len(aggregators)}   class avg: {int(class_avg_now*100)}%"
            cv2.rectangle(frame, (0, 0), (w, 28), _COL_DARK, -1)
            cv2.putText(frame, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COL_WHITE, 1)

            cv2.imshow("EngageBot Classroom Session  [Q = stop]", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

            if time.time() - last_flush >= POST_INTERVAL:
                flush()
                last_flush = time.time()
    finally:
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\n  Stopping session …")
        flush(final=True)
        api.close()
        print("  Done.\n")


if __name__ == "__main__":
    main()
