"""
Phase 2 validation: does the ArcFace embedding actually recognize faces?

Usage:
    python face_match_test.py path/to/photo1.jpg path/to/photo2.jpg

NOTE on thresholds: the w600k_mbf model sits on a higher absolute cosine-
similarity scale than generic ArcFace lore suggests -- validated on real
photos: same person ~0.76, different people ~0.49-0.52. There IS a real
separation margin (~0.25), but no single global constant (like "0.5 = same")
cleanly splits same/different. For actual classroom matching, use RELATIVE
best-match instead: compare a detected face against all enrolled students in
the class and pick the highest-scoring one, with a minimum margin over the
second-best as the "unrecognized" fallback -- not a fixed threshold. The
numbers below are just rough eyeballing for this standalone CLI tool, not
what the real matching logic should use.
"""
import sys

import cv2
import mediapipe as mp

from face_embedding import FaceEmbedder, cosine_similarity, five_points_from_mediapipe


def embed_largest_face(embedder, face_mesh, image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: could not read image: {image_path}")
        sys.exit(1)
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        print(f"ERROR: no face detected in: {image_path}")
        sys.exit(1)
    lm = result.multi_face_landmarks[0].landmark
    pts = five_points_from_mediapipe(lm, w, h)
    emb = embedder.embed_from_frame(img, pts)
    if emb is None:
        print(f"ERROR: alignment failed for: {image_path}")
        sys.exit(1)
    return emb


def main():
    if len(sys.argv) != 3:
        print("Usage: python face_match_test.py photo1.jpg photo2.jpg")
        sys.exit(1)

    embedder = FaceEmbedder()
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)

    e1 = embed_largest_face(embedder, face_mesh, sys.argv[1])
    e2 = embed_largest_face(embedder, face_mesh, sys.argv[2])
    sim = cosine_similarity(e1, e2)

    print(f"\nCosine similarity: {sim:.4f}")
    if sim > 0.65:
        print("-> Likely the SAME person")
    elif sim < 0.45:
        print("-> Likely DIFFERENT people")
    else:
        print("-> Ambiguous (borderline) -- real matching should use relative best-match, not this constant")

    face_mesh.close()


if __name__ == "__main__":
    main()
