"""
Phase 2 validation: live multi-face recognition against enrolled students.

Fetches a class's enrolled face embeddings from the backend, then runs the
webcam loop (same multi-face detection as Phase 1) and labels each detected
face with the best-matching enrolled student name, or "Unknown" if no
confident match (see face_recognizer.py for the matching rule).

Does NOT persist identity across frames yet (no tracker) -- each frame is
recognized independently. That's the next step (custom IoU/centroid
tracker) once this recognition step itself is confirmed working.

Usage:
    python recognize_test.py
"""
import os
import sys
import time

import cv2
import httpx
import mediapipe as mp
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from droid import _LEFT_EYE, _RIGHT_EYE, CAMERA_SOURCE as DEFAULT_CAMERA_SOURCE
from face_embedding import FaceEmbedder, five_points_from_mediapipe
from face_recognizer import FaceRecognizer

API_BASE         = os.getenv("API_BASE", "http://localhost:5000/api")
CAMERA_SOURCE    = os.getenv("CAMERA_SOURCE", DEFAULT_CAMERA_SOURCE)
DEFAULT_EMAIL    = os.getenv("DROID_EMAIL", "")
DEFAULT_PASSWORD = os.getenv("DROID_PASSWORD", "")
MAX_FACES        = int(os.getenv("MAX_FACES", "25"))

_COL_GREEN = (50, 205, 120)
_COL_GRAY  = (130, 130, 130)
_COL_WHITE = (255, 255, 255)
_COL_DARK  = (30, 30, 30)


def _ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else str(default) if default is not None else ""


def fetch_class_recognizer():
    client = httpx.Client(base_url=API_BASE, timeout=10)
    email    = _ask("Teacher email",    DEFAULT_EMAIL    or None)
    password = _ask("Teacher password", DEFAULT_PASSWORD or None)
    r = client.post("/auth/teacher/login", json={"email": email, "password": password})
    r.raise_for_status()
    client.headers.update({"Authorization": f"Bearer {r.json()['token']}"})

    profile = client.get("/teachers/profile"); profile.raise_for_status()
    schedules = client.get("/schedules", params={"teacherId": profile.json()["id"]}).json()
    if not schedules:
        print("No schedules found.")
        sys.exit(1)

    print("\nClasses:")
    for i, s in enumerate(schedules, 1):
        print(f"  [{i}]  {s['classGroup']}  ·  {s['subject']}")
    idx = int(_ask("Select class number", 1)) - 1
    class_group = schedules[idx]["classGroup"]

    r = client.get(f"/students/class/{class_group}/face-embeddings")
    r.raise_for_status()
    enrolled = r.json()
    client.close()

    print(f"\n{class_group}: {len(enrolled)} student(s) enrolled.")
    if not enrolled:
        print("No one enrolled yet in this class -- run enroll_student.py first.")
        sys.exit(1)

    import numpy as np
    for e in enrolled:
        e["embeddings"] = [np.array(v, dtype=np.float32) for v in e["embeddings"]]
    return FaceRecognizer(enrolled)


def _face_bbox(lm, w, h, pad=0.15):
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    x0, x1 = min(xs) * w, max(xs) * w
    y0, y1 = min(ys) * h, max(ys) * h
    pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
    return int(x0 - pw), int(y0 - ph), int(x1 + pw), int(y1 + ph)


def main():
    recognizer = fetch_class_recognizer()
    embedder = FaceEmbedder()

    camera_source = int(CAMERA_SOURCE) if str(CAMERA_SOURCE).isdigit() else CAMERA_SOURCE
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"ERROR: could not open camera source ({camera_source}).")
        sys.exit(1)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=MAX_FACES, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    print("\nPress Q to quit.\n")
    fps_t0, fps_n, fps = time.time(), 0, 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed -- stopping.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                for lm_obj in result.multi_face_landmarks:
                    lm = lm_obj.landmark
                    bbox = _face_bbox(lm, w, h)
                    pts5 = five_points_from_mediapipe(lm, w, h)
                    emb = embedder.embed_from_frame(frame, pts5)

                    if emb is not None:
                        match = recognizer.recognize(emb)
                    else:
                        match = None

                    x0, y0, x1, y1 = bbox
                    if match:
                        col = _COL_GREEN
                        label = f"{match.name}  {match.score:.2f}"
                    else:
                        col = _COL_GRAY
                        label = "Unknown"

                    cv2.rectangle(frame, (x0, y0), (x1, y1), col, 2)
                    cv2.rectangle(frame, (x0, y0 - 22), (x0 + 9 * len(label), y0), _COL_DARK, -1)
                    cv2.putText(frame, label, (x0 + 2, y0 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            fps_n += 1
            if time.time() - fps_t0 >= 1.0:
                fps = fps_n / (time.time() - fps_t0)
                fps_t0, fps_n = time.time(), 0

            cv2.rectangle(frame, (0, 0), (w, 28), _COL_DARK, -1)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COL_WHITE, 1)

            cv2.imshow("Phase 2: Recognition Test  [Q = quit]", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
