"""
Phase 2: matches a live face embedding against a class's enrolled students.

Uses RELATIVE best-match, not a fixed global threshold (see face_match_test.py
for why: the w600k_mbf model's absolute similarity scale doesn't split cleanly
at any single constant). For each probe embedding, this scores it against every
enrolled reference vector, keeps the best score per student, then only accepts
the top student if their score clears MIN_SCORE *and* beats the runner-up by
MIN_MARGIN -- otherwise the face is reported as unrecognized rather than
guessed.
"""
from dataclasses import dataclass
from typing import Optional

from face_embedding import cosine_similarity

MIN_SCORE  = 0.60   # top candidate must at least clear this to be considered
MIN_MARGIN = 0.05   # ...and beat the second-best candidate by at least this much


@dataclass
class Match:
    student_id: str
    name: str
    score: float
    margin: float


class FaceRecognizer:
    def __init__(self, enrolled):
        """`enrolled` is the API response from
        GET /students/class/:classGroup/face-embeddings:
        [{id, name, embeddings: [[float, ...], ...]}, ...]
        """
        self._enrolled = [
            e for e in enrolled if e.get("embeddings")
        ]

    def is_empty(self):
        return len(self._enrolled) == 0

    def recognize(self, probe_embedding) -> Optional[Match]:
        """Returns the best Match if confident, else None (unrecognized)."""
        scored = []
        for student in self._enrolled:
            best_for_student = max(
                cosine_similarity(probe_embedding, ref) for ref in student["embeddings"]
            )
            scored.append((best_for_student, student))

        if not scored:
            return None

        scored.sort(key=lambda t: t[0], reverse=True)
        top_score, top_student = scored[0]
        runner_up_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = top_score - runner_up_score

        if top_score < MIN_SCORE or margin < MIN_MARGIN:
            return None

        return Match(
            student_id=top_student["id"],
            name=top_student["name"],
            score=top_score,
            margin=margin,
        )
