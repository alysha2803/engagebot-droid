"""
Face recognition embeddings via a standalone ArcFace ONNX model (w600k_mbf,
from InsightFace's buffalo_sc pack: github.com/deepinsight/insightface).

Deliberately does NOT depend on the `insightface` PyPI package -- its
`model_zoo` module pulls in the `onnx` package, which requires a newer
protobuf than `mediapipe` tolerates, breaking mediapipe.FaceMesh at
runtime. Using onnxruntime directly avoids that conflict entirely while
still using the same official model weights.

This module only computes embeddings from a face crop + 5 landmark points
(left eye, right eye, nose, mouth-left, mouth-right) that the existing
MediaPipe pipeline already extracts -- it does no detection of its own.
"""
import os

import cv2
import numpy as np
import onnxruntime as ort

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "w600k_mbf.onnx")

# Standard ArcFace 112x112 alignment template (5 points: left eye, right eye,
# nose, mouth-left, mouth-right -- subject's own left/right). Published and
# reused identically across the ArcFace/InsightFace ecosystem.
_ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

_INPUT_SIZE = (112, 112)


class FaceEmbedder:
    def __init__(self, model_path=_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ArcFace model not found at {model_path}. "
                "Download it from https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip "
                "and extract w600k_mbf.onnx into engagebot-droid/models/."
            )
        self._sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name

    def align(self, frame_bgr, five_points):
        """Warps the face in `frame_bgr` to the canonical 112x112 ArcFace pose
        using a similarity transform estimated from `five_points`
        (list of 5 (x, y) tuples: left eye, right eye, nose, mouth-L, mouth-R)."""
        src = np.array(five_points, dtype=np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(src, _ARCFACE_TEMPLATE, method=cv2.LMEDS)
        if matrix is None:
            return None
        return cv2.warpAffine(frame_bgr, matrix, _INPUT_SIZE, borderValue=0.0)

    def embed(self, aligned_face_bgr):
        """Returns a 512-d, L2-normalized embedding for an already-aligned 112x112 BGR crop."""
        rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 127.5
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]   # HWC -> NCHW
        out = self._sess.run(None, {self._input_name: blob})[0][0]
        return out / (np.linalg.norm(out) + 1e-8)

    def embed_from_frame(self, frame_bgr, five_points):
        """Convenience: align + embed in one call. Returns None if alignment fails."""
        aligned = self.align(frame_bgr, five_points)
        if aligned is None:
            return None
        return self.embed(aligned)


def cosine_similarity(a, b):
    return float(np.dot(a, b))   # already L2-normalized, so dot product == cosine similarity


def five_points_from_mediapipe(lm, w, h):
    """Derives the 5 ArcFace alignment points from MediaPipe FaceMesh landmarks,
    reusing the same eye/mouth/nose indices droid.py already uses for EAR/MAR/pose."""
    from droid import _LEFT_EYE, _RIGHT_EYE, _MOUTH_LEFT, _MOUTH_RIGHT

    def _avg(ids):
        xs = [lm[i].x * w for i in ids]
        ys = [lm[i].y * h for i in ids]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    left_eye  = _avg(_LEFT_EYE)
    right_eye = _avg(_RIGHT_EYE)
    nose      = (lm[4].x * w, lm[4].y * h)
    mouth_l   = (lm[_MOUTH_LEFT].x * w, lm[_MOUTH_LEFT].y * h)
    mouth_r   = (lm[_MOUTH_RIGHT].x * w, lm[_MOUTH_RIGHT].y * h)
    return [left_eye, right_eye, nose, mouth_l, mouth_r]
