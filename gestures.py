# gestures.py - hand/face gestures that aren't signs: the summoning palm slap and the
# Sharingan eye cover. Pure timing logic fed (time, position) samples, so it is testable
# and doesn't depend on the frame rate.
from collections import deque

import numpy as np


class PalmSlap:
    """Fires when a hand drops fast into the bottom of the frame (slapping the ground).
    Feed it the raw lowest hand-centre y each frame (None when no hand is seen)."""

    def __init__(self, frame_h, drop=0.2, window=0.25, low=0.55):
        # shortcut: thresholds are guesses (20% of frame height in a quarter second, ending in
        # the bottom 45%), not yet tuned on a webcam; tune here if slaps are missed
        self.h, self.drop, self.window, self.low = frame_h, drop, window, low
        self.samples = deque()

    def update(self, y, now):
        if y is None:   # the hand often leaves the frame mid-slap: judge on the last sample seen
            return False
        self.samples.append((now, y))
        while now - self.samples[0][0] > self.window:
            self.samples.popleft()
        top = min(sy for _, sy in self.samples)
        return y > self.low * self.h and y - top >= self.drop * self.h


def iris(points, idx):
    """(centre, radius) of one iris from face-mesh points (pixels, refine_landmarks=True) and
    that iris' landmark indices."""
    ring = np.asarray(points, float)[list(idx)]
    centre = ring.mean(axis=0)
    return centre, float(np.linalg.norm(ring - centre, axis=1).mean())


class EyeCover:
    """Sharingan trigger: fires once when a hand that covered an eye for hold..max_hold seconds
    is lowered (nothing near the eyes for `lower` seconds). Needs a fresh cover to fire again."""

    def __init__(self, hold=0.3, max_hold=3.0, lower=0.15, reach=0.6):
        # shortcut: thresholds are guesses, not yet tuned on a webcam
        self.hold, self.max_hold, self.lower, self.reach = hold, max_hold, lower, reach
        self.since = self.last = None   # first and last time the eye was seen covered

    def covering(self, eyes, palms):
        """eyes: two iris centres; palms: palm centres. True if a palm sits on either eye."""
        gap = np.linalg.norm(np.subtract(eyes[0], eyes[1]))
        return any(np.linalg.norm(np.subtract(palm, eye)) < self.reach * gap for palm in palms for eye in eyes)

    def update(self, covered, now):
        if covered:
            self.since = now if self.since is None else self.since
            self.last = now
            return False
        if self.since is None or now - self.last < self.lower:   # brief detection drops don't count
            return False
        held, self.since = self.last - self.since, None
        return self.hold <= held <= self.max_hold
