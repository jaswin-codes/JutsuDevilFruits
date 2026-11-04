# sequence.py
import time
from collections import deque

class SequenceDetector:
    def __init__(self, max_length=3, timeout=2.0):
        """
        timeout: seconds allowed between one sign disappearing and the next
                 appearing before the sequence is discarded
        """
        self.sequence = deque(maxlen=max_length)
        self.last_sign = None
        self.timeout = timeout
        self.last_seen = None

    def update(self, stable_sign, now=None):
        """
        Only adds a sign to the sequence when it CHANGES.
        Returns current sequence as a list.
        """
        now = time.monotonic() if now is None else now
        if stable_sign and stable_sign != self.last_sign:
            if self.last_seen is not None and now - self.last_seen > self.timeout:
                self.sequence.clear()   # previous signs are too old to be part of this combo
            self.sequence.append(stable_sign)
            self.last_sign = stable_sign
        if stable_sign:
            self.last_seen = now   # holding a sign doesn't count toward the timeout
        return list(self.sequence)

    def reset(self):
        self.sequence.clear()
        self.last_sign = None
        self.last_seen = None
