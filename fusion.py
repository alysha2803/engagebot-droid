"""
Phase 5: formalizes the scoring that was built incrementally across
Phases 1-4 into one documented, single source of truth, instead of
constants scattered across droid.py/body_cues.py/classroom_session.py.

Two distinct fusion steps, matching the two distinct questions the slides'
"Cues for Engagement Detection" + whole-class summary ask:

1. PER-STUDENT fusion (face + body -> one score):
       final_score = FACE_WEIGHT * face_score + (1 - FACE_WEIGHT) * body_score
   A simple weighted blend, not a trained model -- consistent with every
   other scoring step in this project (droid.py's pose x expr matrix,
   body_cues.py's posture matrix) being an interpretable lookup/blend
   rather than a black box. FACE_WEIGHT defaults to 0.6 because the face
   cues (EAR/MAR/head pose) have actually been validated against real
   behaviour across Phases 1-2, while body cues are newer and less proven
   -- so face stays primary until there's reason to trust body equally.
   Noise is deliberately NOT part of this -- see audio_cues.py for why.

2. CLASS-level summary (per-student scores -> one whole-class picture):
   A plain average of per-student scores hides a bimodal class (half the
   room fully engaged, half checked out averages out to "medium", which is
   misleading). class_engagement_summary() reports the breakdown -- what
   fraction of students are at each level -- not just the mean, so a
   teacher (or the Phase 6 LED colour) sees the actual shape of the room.
"""
from droid import score_to_level

FACE_WEIGHT = 0.6   # face vs body blend weight for fuse_student_score()


def fuse_student_score(face_score, body_score, face_weight=FACE_WEIGHT):
    """Weighted blend of a student's face-cue score and body-cue score."""
    return face_weight * face_score + (1 - face_weight) * body_score


def class_engagement_summary(student_avg_scores):
    """`student_avg_scores` is a list of each currently-tracked student's own
    average final_score (0-1) for the session so far.

    Returns the whole-class picture: not just the mean, but how many
    students sit at each level, so a flat "medium" average doesn't hide a
    class that's actually split between fully-engaged and fully-checked-out.
    """
    if not student_avg_scores:
        return {"avgScore": 0, "level": "low", "breakdown": {"high": 0, "medium": 0, "low": 0}, "studentCount": 0}

    levels = [score_to_level(s) for s in student_avg_scores]
    n = len(levels)
    breakdown = {
        "high":   round(levels.count("high") / n * 100),
        "medium": round(levels.count("medium") / n * 100),
        "low":    round(levels.count("low") / n * 100),
    }
    avg = sum(student_avg_scores) / n
    return {
        "avgScore": round(avg * 100),
        "level": score_to_level(avg),
        "breakdown": breakdown,
        "studentCount": n,
    }
