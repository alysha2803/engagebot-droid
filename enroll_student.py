"""
Phase 2: enroll a student's face so the droid can recognize them later.

Captures a few reference photos (webcam or existing image files), embeds each
via ArcFace, and POSTs the vectors to the backend
(POST /api/students/:id/face-embeddings). Multiple photos per student improve
match robustness across angle/lighting -- aim for 3-5.

Usage:
    python enroll_student.py                      # webcam capture, SPACE to snap, ESC when done
    python enroll_student.py photo1.jpg photo2.jpg # enroll from existing image files instead

Shares login/API config with droid.py (same .env).
"""
import os
import sys

import cv2
import httpx
import mediapipe as mp
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_embedding import FaceEmbedder, five_points_from_mediapipe

API_BASE         = os.getenv("API_BASE", "http://localhost:5000/api")
CAMERA_SOURCE    = os.getenv("CAMERA_SOURCE", "0")
DEFAULT_EMAIL    = os.getenv("DROID_EMAIL", "")
DEFAULT_PASSWORD = os.getenv("DROID_PASSWORD", "")


def _ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else str(default) if default is not None else ""


class EnrollAPI:
    def __init__(self):
        self._client = httpx.Client(base_url=API_BASE, timeout=10)

    def login(self, email, password):
        r = self._client.post("/auth/teacher/login", json={"email": email, "password": password})
        r.raise_for_status()
        self._client.headers.update({"Authorization": f"Bearer {r.json()['token']}"})

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

    def enroll_face(self, student_id, embeddings):
        r = self._client.post(
            f"/students/{student_id}/face-embeddings",
            json={"embeddings": [e.tolist() for e in embeddings]},
        )
        r.raise_for_status()
        return r.json()

    def close(self):
        self._client.close()


def pick_student(api):
    email    = _ask("Teacher email",    DEFAULT_EMAIL    or None)
    password = _ask("Teacher password", DEFAULT_PASSWORD or None)
    print("Logging in … ", end="", flush=True)
    api.login(email, password)
    print("OK")

    profile = api.profile()
    schedules = api.schedules(profile["id"])
    if not schedules:
        print("No schedules found for this teacher.")
        sys.exit(1)

    print("\nClasses:")
    for i, s in enumerate(schedules, 1):
        print(f"  [{i}]  {s['classGroup']}  ·  {s['subject']}")
    idx = int(_ask("Select class number", 1)) - 1
    class_group = schedules[idx]["classGroup"]

    students = api.students(class_group)
    if not students:
        print(f"No students found in {class_group}.")
        sys.exit(1)

    print(f"\nStudents in {class_group}:")
    for i, s in enumerate(students, 1):
        enrolled = " (already enrolled)" if s.get("faceEnrolledAt") else ""
        print(f"  [{i}]  {s.get('name', 'Unknown')}{enrolled}")
    sidx = int(_ask("Select student to enroll", 1)) - 1
    student = students[sidx]
    return student.get("id") or student.get("_id"), student.get("name")


def largest_face_points(face_mesh, img):
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    best_lm, best_area = None, -1
    for lm_obj in result.multi_face_landmarks:
        lm = lm_obj.landmark
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_area, best_lm = area, lm
    return five_points_from_mediapipe(best_lm, w, h)


def capture_from_files(paths, face_mesh, embedder):
    embeddings = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  SKIP (could not read): {path}")
            continue
        pts = largest_face_points(face_mesh, img)
        if pts is None:
            print(f"  SKIP (no face detected): {path}")
            continue
        emb = embedder.embed_from_frame(img, pts)
        if emb is None:
            print(f"  SKIP (alignment failed): {path}")
            continue
        embeddings.append(emb)
        print(f"  OK: {path}")
    return embeddings


def capture_from_webcam(face_mesh, embedder):
    camera_source = int(CAMERA_SOURCE) if str(CAMERA_SOURCE).isdigit() else CAMERA_SOURCE
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"ERROR: could not open camera source ({camera_source}).")
        sys.exit(1)

    print("\nSPACE = capture photo   ESC/Q = finish\n")
    embeddings = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            preview = frame.copy()
            cv2.putText(preview, f"Captured: {len(embeddings)}   SPACE=capture  Q=done",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Enroll Student  [SPACE=capture, Q=done]", preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key == 32:  # SPACE
                pts = largest_face_points(face_mesh, frame)
                if pts is None:
                    print("  No face detected -- try again.")
                    continue
                emb = embedder.embed_from_frame(frame, pts)
                if emb is None:
                    print("  Alignment failed -- try again.")
                    continue
                embeddings.append(emb)
                print(f"  Captured photo {len(embeddings)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return embeddings


def main():
    api = EnrollAPI()
    student_id, student_name = pick_student(api)

    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)
    embedder = FaceEmbedder()

    image_paths = sys.argv[1:]
    if image_paths:
        print(f"\nEnrolling {student_name} from {len(image_paths)} file(s)...")
        embeddings = capture_from_files(image_paths, face_mesh, embedder)
    else:
        print(f"\nEnrolling {student_name} via webcam...")
        embeddings = capture_from_webcam(face_mesh, embedder)

    face_mesh.close()

    if len(embeddings) == 0:
        print("\nNo usable photos captured -- nothing enrolled.")
        sys.exit(1)

    print(f"\nUploading {len(embeddings)} embedding(s) for {student_name}...")
    try:
        resp = api.enroll_face(student_id, embeddings)
        print(f"Done. {resp}")
    except httpx.HTTPStatusError as e:
        print(f"FAILED ({e.response.status_code}): {e.response.text}")
        sys.exit(1)
    finally:
        api.close()


if __name__ == "__main__":
    main()
