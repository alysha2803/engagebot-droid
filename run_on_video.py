"""
Phase 8: runs the detection + tracking + recognition + face/body scoring
pipeline against a RECORDED video file, logging every prediction to a CSV
instead of writing to the database.

This is what makes evaluation repeatable -- a live webcam session can't be
re-run, but a recorded video can be processed again and again (e.g. after
changing a threshold) and compared against the same ground-truth labels
every time.

Scoped to face + body engagement level -- NOT noise. OpenCV's VideoCapture
doesn't read a video's audio track, and noise was never blended into
per-student scores anyway (see audio_cues.py for why). This also matches
the project's own NFR02 (accuracy/F1), which is about engagement level, not
sub-cues like exact head pose -- that's a much harder, more arbitrary thing
for a human to label by eye and isn't what the system is actually judged on.

Timestamps are VIDEO time (frame_index / fps), not wall-clock processing
time -- ground truth labels refer to where you are in the recording, not
how long your laptop took to process it.

Usage:
    python run_on_video.py path/to/recording.mp4 --classgroup "2 Gigih" --out predictions.csv
"""
import argparse
import csv
import os
import sys

import cv2
import httpx
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from droid import (
    _LEFT_EYE, _RIGHT_EYE, _ear, _mar, _head_pose, classify, score_to_level,
)
from face_embedding import FaceEmbedder, five_points_from_mediapipe
from face_recognizer import FaceRecognizer
from tracker import IoUTracker
from body_cues import classify_body
from fusion import fuse_student_score, FACE_WEIGHT

API_BASE         = os.getenv("API_BASE", "http://localhost:5000/api")
DEFAULT_EMAIL    = os.getenv("DROID_EMAIL", "")
DEFAULT_PASSWORD = os.getenv("DROID_PASSWORD", "")
MAX_FACES        = int(os.getenv("MAX_FACES", "25"))
RECOGNITION_INTERVAL = int(os.getenv("RECOGNITION_INTERVAL_FRAMES", "10"))
MIN_FACE_CONTAINMENT = 0.5
# Same quality gate as classroom_session.py, kept in sync so evaluation
# reflects the real live pipeline. Deliberately NOT doing open-set
# auto-enrollment here -- evaluation needs a fixed, reproducible gallery to
# be comparable across runs (e.g. before/after a threshold change).
MIN_FACE_SIZE_FOR_RECOGNITION = int(os.getenv("MIN_FACE_SIZE_FOR_RECOGNITION", "60"))
MAX_YAW_FOR_RECOGNITION       = float(os.getenv("MAX_YAW_FOR_RECOGNITION", "30"))


def _ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else str(default) if default is not None else ""


def fetch_recognizer(class_group):
    client = httpx.Client(base_url=API_BASE, timeout=10)
    email = _ask("Teacher email", DEFAULT_EMAIL or None)
    password = _ask("Teacher password", DEFAULT_PASSWORD or None)
    r = client.post("/auth/teacher/login", json={"email": email, "password": password})
    r.raise_for_status()
    client.headers.update({"Authorization": f"Bearer {r.json()['token']}"})
    r = client.get(f"/students/class/{class_group}/face-embeddings")
    r.raise_for_status()
    enrolled = r.json()
    client.close()
    for e in enrolled:
        e["embeddings"] = [np.array(v, dtype=np.float32) for v in e["embeddings"]]
    print(f"{len(enrolled)} student(s) enrolled in {class_group}.")
    return FaceRecognizer(enrolled)


def _face_bbox(lm, w, h, pad=0.15):
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    x0, x1 = min(xs) * w, max(xs) * w
    y0, y1 = min(ys) * h, max(ys) * h
    pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
    return int(x0 - pw), int(y0 - ph), int(x1 + pw), int(y1 + ph)


def _containment(face_bbox, body_bbox):
    fx0, fy0, fx1, fy1 = face_bbox
    bx0, by0, bx1, by1 = body_bbox
    ix0, iy0 = max(fx0, bx0), max(fy0, by0)
    ix1, iy1 = min(fx1, bx1), min(fy1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    face_area = max(0, fx1 - fx0) * max(0, fy1 - fy0)
    return inter / face_area if face_area > 0 else 0.0


def _match_tracks_to_bodies(tracks, body_boxes):
    pairs = []
    for ti, track in enumerate(tracks):
        for bi, bbox in enumerate(body_boxes):
            c = _containment(track.bbox, bbox)
            if c >= MIN_FACE_CONTAINMENT:
                pairs.append((c, ti, bi))
    pairs.sort(reverse=True)
    assigned, used_t, used_b = {}, set(), set()
    for c, ti, bi in pairs:
        if ti in used_t or bi in used_b:
            continue
        assigned[ti] = bi
        used_t.add(ti)
        used_b.add(bi)
    return assigned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to the recorded video file")
    parser.add_argument("--classgroup", required=True, help="Class group to fetch enrolled students from")
    parser.add_argument("--out", default="predictions.csv", help="Output predictions CSV path")
    parser.add_argument("--use-body", action="store_true", help="Also run YOLOv8-Pose body cues (slower)")
    args = parser.parse_args()

    recognizer = fetch_recognizer(args.classgroup)
    if recognizer.is_empty():
        print(f"No one enrolled in {args.classgroup} -- run enroll_student.py first.")
        sys.exit(1)

    embedder = FaceEmbedder()
    tracker = IoUTracker()
    pose_model = None
    if args.use_body:
        from ultralytics import YOLO
        pose_model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open video file ({args.video}).")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=MAX_FACES, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    rows = []
    frame_i = 0
    print(f"Processing {args.video} ({fps:.1f} fps)... this runs as fast as your CPU allows, not in real time.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_i += 1
            elapsed = frame_i / fps
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            landmark_sets = result.multi_face_landmarks or []
            bboxes = [_face_bbox(lm_obj.landmark, w, h) for lm_obj in landmark_sets]
            tracks = tracker.update(bboxes)

            body_matches = {}
            last_body_kps = []
            if pose_model is not None:
                pose_result = pose_model(frame, verbose=False)[0]
                if pose_result.keypoints is not None and len(pose_result.keypoints.data):
                    last_body_kps = pose_result.keypoints.data.cpu().numpy()
                    body_boxes = [tuple(b) for b in pose_result.boxes.xyxy.cpu().numpy().astype(int)]
                    body_matches = _match_tracks_to_bodies(tracks, body_boxes)

            for ti, (track, lm_obj) in enumerate(zip(tracks, landmark_sets)):
                lm = lm_obj.landmark

                ear_avg = (_ear(lm, _LEFT_EYE, w, h) + _ear(lm, _RIGHT_EYE, w, h)) / 2.0
                mar_val = _mar(lm, w, h)
                pitch, yaw = _head_pose(lm, w, h)
                pose, expr, face_score = classify(pitch, yaw, ear_avg, mar_val)

                tx0, ty0, tx1, ty1 = track.bbox
                face_quality_ok = (
                    min(tx1 - tx0, ty1 - ty0) >= MIN_FACE_SIZE_FOR_RECOGNITION
                    and abs(yaw) <= MAX_YAW_FOR_RECOGNITION
                )

                if track.frames_since_recognition >= RECOGNITION_INTERVAL and face_quality_ok:
                    pts5 = five_points_from_mediapipe(lm, w, h)
                    emb = embedder.embed_from_frame(frame, pts5)
                    track.frames_since_recognition = 0
                    if emb is not None:
                        match = recognizer.recognize(emb)
                        if match:
                            track.register_vote(match.student_id, match.name)
                        # No match and no auto-enrollment here -- track.student_id
                        # simply stays whatever it already was (majority vote so far).

                if not track.student_id:
                    continue

                posture = None
                final_score = face_score
                if ti in body_matches:
                    kps = last_body_kps[body_matches[ti]]
                    posture, body_score, shoulder_w = classify_body(kps, track.body_baseline)
                    if shoulder_w and posture == "neutral":
                        track.body_baseline = shoulder_w if track.body_baseline is None else (0.9 * track.body_baseline + 0.1 * shoulder_w)
                    final_score = fuse_student_score(face_score, body_score, FACE_WEIGHT)

                rows.append({
                    "elapsed_seconds": round(elapsed, 2),
                    "student_id": track.student_id,
                    "student_name": track.student_name,
                    "pose": pose,
                    "expr": expr,
                    "posture": posture or "",
                    "score": round(final_score, 4),
                    "level": score_to_level(final_score),
                })

            if frame_i % 300 == 0:
                print(f"  ...{elapsed:.0f}s processed ({frame_i} frames)")
    finally:
        face_mesh.close()
        cap.release()

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "elapsed_seconds", "student_id", "student_name",
            "pose", "expr", "posture", "score", "level",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} prediction rows to {args.out}")


if __name__ == "__main__":
    main()
