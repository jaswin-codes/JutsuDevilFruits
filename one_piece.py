# one_piece.py - One Piece mode: Luffy's Gomu Gomu rubber body over the webcam image.
#   Fist:  Gomu Gomu no Pistol - the fist shoots out the way the hand points, the arm stretching
#          behind it; open the hand and it snaps back.
#   Pinch: grab your face (mouth corner, cheek, ear...) between thumb and index finger and pull:
#          the skin stretches out past your fingers; let go and it snaps back with a wobble.
# Only hand landmarks are needed. stretch/Spring/gesture/Gomu are pure (no camera) and tested.
import math
import time
import cv2
import numpy as np
from effects import play_sound
from hand_detector import HandDetector

PISTOL = 6.0             # fist reach, in palm sizes (wrist to middle knuckle); the frame edge also stops it
PULL_GAIN = 2.0          # pinch pull: the skin travels this many times as far as your fingers
PINCH_ON, PINCH_OFF = 0.3, 0.45   # thumb-index tip gap in palm sizes: grab below ON, let go above OFF
STIFF, DAMP = 300.0, 12.0         # rubber spring: underdamped, so a release snaps back with a wobble
MAX_DT = 0.1


def stretch(frame, grab, tip, radius, back=None):
    """Rubber stretch, in place: what is around `grab` is moved to `tip`, and the strip from
    `back` pixels behind grab (away from tip) up to grab is drawn stretched over the whole gap,
    inside a band of `radius` pixels. Nothing behind that strip moves, so there is no fold."""
    G, T = np.float32(grab), np.float32(tip)
    gap = float(np.hypot(*(T - G)))
    if gap < 1:
        return
    back = radius if back is None else back
    u = (T - G) / gap
    n = np.float32([-u[1], u[0]])
    A = G - back * u                          # anchor: the start of the stretched strip
    length = back + gap                       # anchor to tip on screen
    h, w = frame.shape[:2]
    x0, y0 = np.maximum(np.floor(np.minimum(A, T) - radius), 0).astype(int)
    x1, y1 = np.minimum(np.ceil(np.maximum(A, T) + radius) + 1, (w, h)).astype(int)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    qx, qy = xs - A[0], ys - A[1]
    a, b = qx * u[0] + qy * u[1], qx * n[0] + qy * n[1]      # along / across the band
    src = np.where(a <= 0, a, np.where(a <= length, a * (back / length), a - gap))
    patch = cv2.remap(frame, A[0] + src * u[0] + b * n[0], A[1] + src * u[1] + b * n[1],
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    dist = np.hypot(a - np.clip(a, 0, length), b)            # distance to the band's centre line
    alpha = np.clip((1 - dist / radius) * 3, 0, 1)[..., None]   # soft outer third
    roi = frame[y0:y1, x0:x1]
    roi[:] = (patch * alpha + roi * (1 - alpha)).astype(np.uint8)


class Spring:
    """2D underdamped spring: hold() follows a point directly (keeping its speed), step() pulls toward a target."""

    def __init__(self):
        self.x, self.v = np.zeros(2), np.zeros(2)

    def hold(self, x, dt):
        x = np.asarray(x, float)
        self.v = (x - self.x) / max(dt, 1e-3)
        self.x = x

    def step(self, target, dt):
        h = min(dt, MAX_DT) / 4                  # substeps keep the stiff spring stable
        for _ in range(4):
            self.v += (-STIFF * (self.x - target) - DAMP * self.v) * h
            self.x += self.v * h


def _d(p, q):
    return math.dist(p, q)


def gesture(lm, pinching=False):
    """'fist', 'pinch' or None from 21 hand landmarks in pixels. pinching: last frame's answer
    was 'pinch' (looser release threshold, so a grab doesn't flicker)."""
    if all(_d(lm[t], lm[0]) < _d(lm[p], lm[0]) for t, p in ((8, 6), (12, 10), (16, 14), (20, 18))):
        return "fist"   # checked first: in a fist the thumb rests on the index finger, like a pinch
    palm = max(_d(lm[0], lm[9]), 1.0)
    middle_out = _d(lm[12], lm[0]) > _d(lm[10], lm[0])
    if middle_out and _d(lm[4], lm[8]) < (PINCH_OFF if pinching else PINCH_ON) * palm:
        return "pinch"
    return None


class Grip:
    """One hand's stretch. spring.x is the tip offset from src."""

    def __init__(self, wrist):
        self.wrist = wrist
        self.kind = self.shape = None   # current gesture / gesture whose stretch is still drawn
        self.src, self.palm, self.dir = None, 40.0, np.zeros(2)
        self.spring = Spring()


class Gomu:
    def __init__(self, frame_w, frame_h):
        self.w, self.h = frame_w, frame_h
        self.grips = []

    def _reach(self, start, direction, palm):
        """Fist travel along direction, stopped so the fist stays in the frame."""
        limits = [PISTOL * palm]
        for p, d, size in zip(start, direction, (self.w, self.h)):
            if abs(d) > 1e-6:
                limits.append(((size - 1 - p) if d > 0 else p) / abs(d) - palm)
        return max(0.0, min(limits))

    def update(self, hands, dt):
        """hands: list of 21-landmark pixel lists. Returns (grab, tip, radius, back) stretches to draw."""
        dt = min(dt, MAX_DT)
        free = list(self.grips)
        for lm in hands:
            grip = min(free, key=lambda g: _d(g.wrist, lm[0]), default=None)
            palm = max(_d(lm[0], lm[9]), 1.0)
            if grip is None or _d(grip.wrist, lm[0]) > 3 * palm:
                grip = Grip(lm[0])
                self.grips.append(grip)
            else:
                free.remove(grip)
            grip.wrist, grip.palm = lm[0], palm
            kind = gesture(lm, grip.kind == "pinch")
            if kind != grip.kind and kind:                    # a new grab: start from scratch
                grip.spring = Spring()
                grip.shape = kind
                if kind == "pinch":
                    grip.src = np.mean([lm[4], lm[8]], axis=0)  # image point under the fingers, fixed
            grip.kind = kind
            if kind == "fist":
                grip.src = np.mean([lm[0], lm[9]], axis=0)       # fist centre (fingers curled to the palm)
                grip.dir = np.subtract(lm[9], lm[0]) / palm
                grip.spring.step(grip.dir * self._reach(grip.src, grip.dir, palm), dt)
            elif kind == "pinch":
                grip.spring.hold(PULL_GAIN * (np.mean([lm[4], lm[8]], axis=0) - grip.src), dt)
        for grip in free:                                     # hand gone: let go
            grip.kind = None
        out = []
        for grip in self.grips:
            if grip.kind is None:
                grip.spring.step(np.zeros(2), dt)
            if grip.shape is None:
                continue
            if grip.kind is None and np.hypot(*grip.spring.x) < 2 and np.hypot(*grip.spring.v) < 30:
                grip.shape = None                             # snapped back and settled
                continue
            fist = grip.shape == "fist"
            radius = grip.palm * (0.9 if fist else 0.45)
            out.append((grip.src, grip.src + grip.spring.x, radius, radius * (2.0 if fist else 1.2)))
        self.grips = [g for g in self.grips if g.kind or g.shape or g not in free]
        return out

    def pistol(self):
        """A fist is out far enough to call the attack."""
        return any(g.kind == "fist" and np.hypot(*g.spring.x) > 2 * g.palm for g in self.grips)


def play(cap):
    """One Piece mode loop until Q ("quit") or M ("menu")."""
    detector = HandDetector(max_hands=2)
    gomu, last, was_pistol = None, time.monotonic(), False
    while True:
        success, frame = cap.read()
        if not success:
            print("Camera stopped sending frames.")
            return "quit"
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        gomu = gomu or Gomu(w, h)
        _, hands = detector.find_hands(frame, draw=False)
        now = time.monotonic()
        dt, last = now - last, now
        for grab, tip, radius, back in gomu.update([detector.get_landmark_list(lm, frame.shape) for lm in hands], dt):
            stretch(frame, grab, tip, radius, back)

        pistol = gomu.pistol()
        if pistol and not was_pistol:
            play_sound(250, 700)
        was_pistol = pistol
        if pistol:
            text = "GOMU GOMU NO PISTOL!"
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.3, 3)[0]
            cv2.putText(frame, text, ((w - size[0]) // 2, 60), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 220), 3)
        cv2.putText(frame, "ONE PIECE", (10, h - 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, "Fist: Pistol | Pinch your mouth and pull | M menu, Q quit", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.imshow("Jutsu Code", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return "quit"
        if key == ord('m'):
            return "menu"
