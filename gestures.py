# gestures.py - hand/face gestures that aren't signs: the summoning palm slap and the
# Sharingan eye cover. Pure timing logic fed (time, position) samples, so it is testable
# and doesn't depend on the frame rate.
from collections import deque


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
