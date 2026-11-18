"""
Phase 8 follow-up: synthetic enrollment augmentation for the
"Single Sample Per Person" (SSPP) constraint -- schools can only provide one
official passport-style photo per (underage) student, not multiple candid
photos, for privacy reasons.

Rather than needing more real photos of the student, this generates
multiple SYNTHETIC variants from the ONE already-authorized photo, each
embedded separately and stored as that student's reference set (the same
`faceEmbeddings: [[Number]]` array already supports N vectors per student --
this just fills it from augmented copies of one photo instead of from
multiple real photos). Standard SSPP practice, touches zero new images of
the student.

Augmentation happens on the ALREADY-ALIGNED 112x112 ArcFace crop, not the
raw photo -- simpler and more robust than re-running face detection on each
synthetic full-image variant (a geometric transform of the whole photo can
shift or distort the face in ways that make re-detection unreliable).

The transforms are chosen to simulate the actual domain gap between a
passport photo (frontal, neutral, studio lighting, close-up) and a live
classroom frame (angled, varied lighting, some blur/compression) -- small
rotations, brightness/contrast shifts, slight blur, a horizontal flip
(ArcFace models are themselves trained with random flipping, so this is a
valid same-identity variant, not a distortion), and minor crop jitter.
"""
import cv2
import numpy as np

_SIZE = 112


def _rotate(crop, degrees):
    center = (_SIZE / 2, _SIZE / 2)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    return cv2.warpAffine(crop, matrix, (_SIZE, _SIZE), borderMode=cv2.BORDER_REFLECT)


def _brightness_contrast(crop, alpha, beta):
    return cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)


def _blur(crop, ksize):
    return cv2.GaussianBlur(crop, (ksize, ksize), 0)


def _flip(crop):
    return cv2.flip(crop, 1)


def _translate(crop, dx, dy):
    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(crop, matrix, (_SIZE, _SIZE), borderMode=cv2.BORDER_REFLECT)


# Each entry is a (name, function) pair applied to the clean aligned crop.
# Deterministic, not random -- reproducible and easy to reason about/debug.
_AUGMENTATIONS = [
    ("rotate+8",       lambda c: _rotate(c, 8)),
    ("rotate-8",       lambda c: _rotate(c, -8)),
    ("rotate+15",      lambda c: _rotate(c, 15)),
    ("rotate-15",      lambda c: _rotate(c, -15)),
    ("flip",           lambda c: _flip(c)),
    ("flip+rotate-8",  lambda c: _rotate(_flip(c), -8)),
    ("brighter",       lambda c: _brightness_contrast(c, 1.0, 20)),
    ("darker",         lambda c: _brightness_contrast(c, 1.0, -20)),
    ("higher_contrast", lambda c: _brightness_contrast(c, 1.2, 0)),
    ("lower_contrast", lambda c: _brightness_contrast(c, 0.8, 10)),
    ("blur",           lambda c: _blur(c, 3)),
    ("blur+rotate+5",  lambda c: _blur(_rotate(c, 5), 3)),
    ("translate",      lambda c: _translate(c, 4, 3)),
    ("translate2",     lambda c: _translate(c, -4, -3)),
]


def augment_aligned_face(aligned_crop_bgr, n=10):
    """Returns up to `n` synthetic variants of the aligned 112x112 crop
    (the clean original is NOT included -- the caller should embed that
    separately). Pass n=0 or omit to disable augmentation entirely."""
    n = min(n, len(_AUGMENTATIONS))
    return [fn(aligned_crop_bgr) for _, fn in _AUGMENTATIONS[:n]]


def augmentation_names(n=10):
    n = min(n, len(_AUGMENTATIONS))
    return [name for name, _ in _AUGMENTATIONS[:n]]
