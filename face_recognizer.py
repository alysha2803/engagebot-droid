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
    def __init__(self, enrolled, pending=None):
        """`enrolled` is the API response from
        GET /students/class/:classGroup/face-embeddings:
        [{id, name, embeddings: [[float, ...], ...]}, ...]

        `pending` is the equivalent for not-yet-named auto-enrolled faces
        (GET /pending-faces/class/:classGroup/embeddings) -- matched against
        on equal footing with real students, so a face already auto-enrolled
        earlier this class gets recognized as "itself" instead of spawning a
        duplicate pending identity every time it's seen.
        """
        self._gallery = {}   # id -> {"name", "embeddings", "kind"}
        for e in enrolled:
            if e.get("embeddings"):
                self._gallery[e["id"]] = {"name": e["name"], "embeddings": e["embeddings"], "kind": "student"}
        for e in (pending or []):
            if e.get("embeddings"):
                self._gallery[e["id"]] = {"name": e["name"], "embeddings": e["embeddings"], "kind": "pending"}

    def is_empty(self):
        return len(self._gallery) == 0

    def add_candidate(self, candidate_id, name, embeddings, kind="pending"):
        """Registers a newly auto-enrolled face so later frames in the same
        session can match against it immediately, without re-fetching the
        whole gallery from the backend."""
        self._gallery[candidate_id] = {"name": name, "embeddings": list(embeddings), "kind": kind}

    def update_candidate_embeddings(self, candidate_id, embeddings):
        """Keeps the in-memory gallery in sync after a growth PATCH."""
        if candidate_id in self._gallery:
            self._gallery[candidate_id]["embeddings"] = list(embeddings)

    def append_embedding(self, candidate_id, embedding):
        """Appends one more embedding locally, mirroring a successful growth
        PATCH on the backend without needing a full gallery re-fetch."""
        if candidate_id in self._gallery:
            self._gallery[candidate_id]["embeddings"].append(embedding)

    def kind_of(self, candidate_id):
        entry = self._gallery.get(candidate_id)
        return entry["kind"] if entry else None

    def recognize(self, probe_embedding) -> Optional[Match]:
        """Returns the best Match if confident, else None (unrecognized)."""
        scored = []
        for candidate_id, entry in self._gallery.items():
            best_for_candidate = max(
                cosine_similarity(probe_embedding, ref) for ref in entry["embeddings"]
            )
            scored.append((best_for_candidate, candidate_id, entry))

        if not scored:
            return None

        scored.sort(key=lambda t: t[0], reverse=True)
        top_score, top_id, top_entry = scored[0]
        runner_up_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = top_score - runner_up_score

        if top_score < MIN_SCORE or margin < MIN_MARGIN:
            return None

        return Match(
            student_id=top_id,
            name=top_entry["name"],
            score=top_score,
            margin=margin,
        )
