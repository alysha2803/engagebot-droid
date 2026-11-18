"""
Phase 1 validation: multi-face detection + per-face cue extraction.

Does NOT write to the database — no student identity exists yet (that's
Phase 2). This purely proves two things:
  1. MediaPipe can detect and score multiple faces in one frame.
  2. How many faces it can handle while staying usably fast (FPS counter).

NOTE: Face ordering is NOT stable frame-to-frame in MediaPipe's multi-face
mode -- "Face 1" this frame may be a different person next frame. That's
expected here; a real tracker (ByteTrack) is added in Phase 2.

Usage:
    python multi_face_test.py
Env (optional, same .env as droid.py):
    CAMERA_SOURCE           - 0, or a Pi stream URL (default: from .env)
    MAX_FACES               - max faces to detect at once (default: 25)
    PROCESS_EVERY_N_FRAMES  - skip frames to save CPU (default: 1 = every frame)
"""
import os
import sys
import time

import cv2
import mediapipe as mp
from dotenv import load_dotenv

load_dotenv()

# Reuse the proven per-face logic from droid.py -- no duplication.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from droid import (
    _LEFT_EYE, _RIGHT_EYE, _ear, _mar, _head_pose,
    classify, score_to_level, CAMERA_SOURCE as DEFAULT_CAMERA_SOURCE,
)

CAMERA_SOURCE          = os.getenv("CAMERA_SOURCE", DEFAULT_CAMERA_SOURCE)
MAX_FACES              = int(os.getenv("MAX_FACES", "25"))
PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "1"))

_COL_GREEN  = (50,  205, 120)
_COL_ORANGE = (0,   165, 255)
_COL_RED    = (60,   80, 220)
_COL_WHITE  = (255, 255, 255)
_COL_DARK   = (30,   30,  30)

_LEVEL_COLOUR = {"high": _COL_GREEN, "medium": _COL_ORANGE, "low": _COL_RED}


def _face_bbox(lm, w, h, pad=0.15):
    """Bounding box around a landmark set, with a little padding for the label."""
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    x0, x1 = min(xs) * w, max(xs) * w
    y0, y1 = min(ys) * h, max(ys) * h
    pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
    return int(x0 - pw), int(y0 - ph), int(x1 + pw), int(y1 + ph)


def main():
    camera_source = int(CAMERA_SOURCE) if str(CAMERA_SOURCE).isdigit() else CAMERA_SOURCE
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera source ({camera_source}).")
        sys.exit(1)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=MAX_FACES,
        refine_landmarks=False,   # iris landmarks unused here; skipping saves compute
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print(f"MAX_FACES={MAX_FACES}  PROCESS_EVERY_N_FRAMES={PROCESS_EVERY_N_FRAMES}")
    print("Press Q to quit.\n")

    frame_i = 0
    fps_t0, fps_n = time.time(), 0
    fps = 0.0
    last_faces = []   # cached results on skipped frames

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed — stopping.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            frame_i += 1

            if frame_i % PROCESS_EVERY_N_FRAMES == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)
                faces = []
                if result.multi_face_landmarks:
                    for lm_obj in result.multi_face_landmarks:
                        lm = lm_obj.landmark
                        ear_avg = (_ear(lm, _LEFT_EYE, w, h) + _ear(lm, _RIGHT_EYE, w, h)) / 2.0
                        mar_val = _mar(lm, w, h)
                        pitch, yaw = _head_pose(lm, w, h)
                        pose, expr, score = classify(pitch, yaw, ear_avg, mar_val)
                        faces.append({
                            "bbox": _face_bbox(lm, w, h),
                            "pose": pose, "expr": expr, "score": score,
                            "level": score_to_level(score),
                        })
                last_faces = faces

                fps_n += 1
                if time.time() - fps_t0 >= 1.0:
                    fps = fps_n / (time.time() - fps_t0)
                    fps_t0, fps_n = time.time(), 0

            # Draw every frame using the latest known results (even on skipped frames).
            for f in last_faces:
                x0, y0, x1, y1 = f["bbox"]
                col = _LEVEL_COLOUR.get(f["level"], _COL_WHITE)
                cv2.rectangle(frame, (x0, y0), (x1, y1), col, 2)
                label = f"{f['pose']}/{f['expr']}  {int(f['score']*100)}%"
                cv2.rectangle(frame, (x0, y0 - 22), (x0 + 8 * len(label), y0), _COL_DARK, -1)
                cv2.putText(frame, label, (x0 + 2, y0 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            class_avg = (sum(f["score"] for f in last_faces) / len(last_faces)) if last_faces else 0.0
            header = f"Faces: {len(last_faces)}   class avg (preview): {int(class_avg*100)}%   FPS: {fps:.1f}"
            cv2.rectangle(frame, (0, 0), (w, 30), _COL_DARK, -1)
            cv2.putText(frame, header, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COL_WHITE, 1)

            cv2.imshow("Phase 1: Multi-Face Detection Test  [Q = quit]", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
