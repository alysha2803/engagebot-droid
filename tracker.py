"""
Phase 2: lightweight multi-face tracker.

Deliberately NOT YOLO's ByteTrack/BoT-SORT -- detection here comes from
MediaPipe, not YOLO, so pulling in full Ultralytics just for tracking would
be unnecessary weight. This is a simple greedy IoU tracker: each frame's
detections are matched to existing tracks by bounding-box overlap, unmatched
tracks age out after MAX_MISSED frames, unmatched detections become new
tracks.

Why track at all, if we already recognize faces every frame? Recognition
(embedding + matching) is the expensive step -- tracking lets the caller
re-run recognition only occasionally per track (e.g. every N frames) and
otherwise just carry the cached identity forward, while still handling
faces appearing/disappearing/moving between frames.
"""
from dataclasses import dataclass, field
from typing import Optional


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    bbox: tuple
    missed: int = 0
    # Caller-managed identity cache -- the tracker itself doesn't know about
    # students, it just carries whatever the caller stashes here forward.
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    frames_since_recognition: int = 999   # force recognition on first sighting
    score_history: list = field(default_factory=list)


class IoUTracker:
    MIN_IOU = 0.3       # below this, detections don't match an existing track
    MAX_MISSED = 15      # ~0.5s at 30fps before a track is dropped

    def __init__(self):
        self._tracks = {}
        self._next_id = 1

    def update(self, detections):
        """`detections` is a list of bboxes (x0, y0, x1, y1) for this frame.
        Returns the list of Track objects currently alive, in the same order
        as `detections` for matched ones (new tracks appended at the end)."""
        unmatched_dets = list(range(len(detections)))
        matched_tracks = set()
        ordered = [None] * len(detections)

        # Greedy match: best IoU first.
        pairs = []
        for tid, track in self._tracks.items():
            for di in unmatched_dets:
                iou = _iou(track.bbox, detections[di])
                if iou >= self.MIN_IOU:
                    pairs.append((iou, tid, di))
        pairs.sort(reverse=True)

        used_dets, used_tracks = set(), set()
        for iou, tid, di in pairs:
            if di in used_dets or tid in used_tracks:
                continue
            track = self._tracks[tid]
            track.bbox = detections[di]
            track.missed = 0
            track.frames_since_recognition += 1
            used_dets.add(di)
            used_tracks.add(tid)
            matched_tracks.add(tid)
            ordered[di] = track

        # New tracks for unmatched detections.
        for di in range(len(detections)):
            if di in used_dets:
                continue
            track = Track(id=self._next_id, bbox=detections[di])
            self._next_id += 1
            self._tracks[track.id] = track
            ordered[di] = track

        # Age out tracks that weren't matched this frame.
        for tid in list(self._tracks.keys()):
            if tid not in used_tracks:
                self._tracks[tid].missed += 1
                if self._tracks[tid].missed > self.MAX_MISSED:
                    del self._tracks[tid]

        return ordered
