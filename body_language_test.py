"""
Phase 3 validation: multi-person body-language detection via YOLOv8-Pose.

Standalone, like multi_face_test.py was for Phase 1 -- no face recognition,
no tracking, no DB writes. Just proves: can YOLOv8-Pose find everyone's
upper-body keypoints in the frame, and do the body_cues.py heuristics
(hand raised / head down / leaning in-out) look reasonable when you
actually do those things in front of the camera?

First run downloads the yolov8n-pose.pt weights automatically (~6MB) into
this folder.

NOTE: the per-person "baseline" shoulder width here is keyed by YOLO's
result-array index, which is NOT a stable identity across frames (same
caveat as Phase 1's face ordering) -- fine for this single-purpose
detection/heuristic sanity check, not meant to persist identity. Real
per-student baselines will come from the same tracker.py IDs once this is
wired into classroom_session.py.

Usage:
    python body_language_test.py
"""
import os
import sys

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from body_cues import classify_body

CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")

_COL_GREEN  = (50, 205, 120)
_COL_ORANGE = (0, 165, 255)
_COL_RED    = (60, 80, 220)
_COL_WHITE  = (255, 255, 255)
_COL_DARK   = (30, 30, 30)
_POSTURE_COLOUR = {
    "hand_raised": _COL_GREEN,
    "leaning_in":  _COL_GREEN,
    "neutral":     _COL_ORANGE,
    "head_down":   _COL_RED,
    "slouched":    _COL_RED,
}

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),         # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # shoulders/arms
]


def main():
    camera_source = int(CAMERA_SOURCE) if str(CAMERA_SOURCE).isdigit() else CAMERA_SOURCE
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"ERROR: could not open camera source ({camera_source}).")
        sys.exit(1)

    model = YOLO("yolov8n-pose.pt")
    baselines = {}   # naive per-slot shoulder-width baseline (no tracker here yet)

    print("Press Q to quit.\n")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed -- stopping.")
                break
            frame = cv2.flip(frame, 1)

            results = model(frame, verbose=False)[0]
            if results.keypoints is not None:
                all_kps = results.keypoints.data.cpu().numpy()   # (N, 17, 3)
                boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []

                for i, kps in enumerate(all_kps):
                    baseline = baselines.get(i)
                    posture, score, shoulder_w = classify_body(kps, baseline)
                    # Only drift the baseline toward "resting" (neutral) widths --
                    # if we also updated it while actively leaning, the baseline
                    # would chase the lean and erase the deviation within ~1s.
                    if shoulder_w and posture == "neutral":
                        baselines[i] = shoulder_w if baseline is None else (0.9 * baseline + 0.1 * shoulder_w)

                    col = _POSTURE_COLOUR.get(posture, _COL_WHITE)

                    for a, b in SKELETON_EDGES:
                        xa, ya, ca = kps[a]
                        xb, yb, cb = kps[b]
                        if ca >= 0.3 and cb >= 0.3:
                            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), col, 2)
                    for x, y, c in kps[:11]:
                        if c >= 0.3:
                            cv2.circle(frame, (int(x), int(y)), 4, col, -1)

                    if i < len(boxes):
                        x0, y0, x1, y1 = boxes[i].astype(int)
                        label = f"{posture}  {int(score*100)}%"
                        cv2.rectangle(frame, (x0, y0 - 22), (x0 + 9 * len(label), y0), _COL_DARK, -1)
                        cv2.putText(frame, label, (x0 + 2, y0 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            cv2.imshow("Phase 3: Body Language Test  [Q = quit]", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
