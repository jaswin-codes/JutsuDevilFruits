# one_piece.py - One Piece mode: Luffy's Gomu Gomu rubber body over the webcam image.
#   Fist:  Gomu Gomu no Pistol - the fist shoots out the way the hand points, the arm stretching
#          behind it; open the hand and it snaps back.
#          Pump the fist back and forth fast: Gomu Gomu no Gatling, a blur of fists.
#   V sign (index + middle up), held: Gear Second for GEAR_TIME s - pink steaming body, faster Pistol.
#   Pinch: grab your face (mouth corner, cheek, ear...) between thumb and index finger and pull:
#          the skin stretches out past your fingers; let go and it snaps back with a wobble.
# Only hand landmarks are needed. stretch/Spring/gesture/Gomu are pure (no camera) and tested.
import math
import time
import cv2
import mediapipe as mp
import numpy as np
from effects import ParticleSystem, play_sound
from hand_detector import HandDetector

PISTOL = 6.0             # fist reach, in palm sizes (wrist to middle knuckle); the frame edge also stops it
PULL_GAIN = 2.0          # pinch pull: the skin travels this many times as far as your fingers
PINCH_ON, PINCH_OFF = 0.3, 0.45   # thumb-index tip gap in palm sizes: grab below ON, let go above OFF
STIFF, DAMP = 300.0, 12.0         # rubber spring: underdamped, so a release snaps back with a wobble
MAX_DT = 0.1
GRACE = 3                # frames a lost hand or misread gesture keeps its grab (tracking drops a frame often)
PUMP_DIST = 0.45         # Gatling: a pump counts once the wrist has travelled this many palm sizes (beats jitter)
GATLING_PUMPS, GATLING_WINDOW = 4, 1.0   # this many direction reversals within this many seconds
GATLING_HOLD = 0.5       # seconds the Gatling keeps going after the last pump
GATLING_FISTS = 6        # extra fists drawn around the main one
GATLING_SPREAD = 0.45    # radians either side of the fist direction
GEAR_HOLD = 0.5          # seconds a gear sign must be held to switch the gear on
GEAR_TIME = 10.0         # seconds a gear lasts
GEAR2_BOOST = 4.0        # Gear Second spring stiffness factor: Pistol is sqrt(4) = 2x faster
GEAR2_TINT = 0.35        # how pink the body gets (0..1)


def stretch(frame, grab, tip, radius, back=None, image=None):
    """Rubber stretch, in place: what is around `grab` is moved to `tip`, and the strip from
    `back` pixels behind grab (away from tip) up to grab is drawn stretched over the whole gap,
    inside a band of `radius` pixels. Nothing behind that strip moves, so there is no fold.
    image: what to read from (default frame), so several stretches of one grab copy the clean fist."""
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
    patch = cv2.remap(frame if image is None else image, A[0] + src * u[0] + b * n[0], A[1] + src * u[1] + b * n[1],
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

    def step(self, target, dt, boost=1.0):
        """boost scales stiffness; damping by its root keeps the same wobble, just faster."""
        h = min(dt, MAX_DT) / 4                  # substeps keep the stiff spring stable
        for _ in range(4):
            self.v += (-STIFF * boost * (self.x - target) - DAMP * math.sqrt(boost) * self.v) * h
            self.x += self.v * h


def _d(p, q):
    return math.dist(p, q)


def gesture(lm, pinching=False):
    """'fist', 'pinch', 'two' (V sign) or None from 21 hand landmarks in pixels. pinching: last
    frame's answer was 'pinch' (looser release threshold, so a grab doesn't flicker)."""
    out = [_d(lm[t], lm[0]) > _d(lm[p], lm[0]) for t, p in ((8, 6), (12, 10), (16, 14), (20, 18))]
    if not any(out):
        return "fist"   # checked first: in a fist the thumb rests on the index finger, like a pinch
    palm = max(_d(lm[0], lm[9]), 1.0)
    if out[1] and _d(lm[4], lm[8]) < (PINCH_OFF if pinching else PINCH_ON) * palm:
        return "pinch"
    if out == [True, True, False, False]:
        return "two"
    return None


GEARS = {"two": 2}            # gear sign -> gear


class Grip:
    """One hand's stretch. spring.x is the tip offset from src."""

    def __init__(self, wrist):
        self.wrist = wrist
        self.kind = self.shape = None   # current gesture / gesture whose stretch is still drawn
        self.src, self.palm, self.dir = None, 40.0, np.zeros(2)
        self.spring = Spring()
        self.missed = 0                 # frames in a row the gesture was lost
        self.anchor, self.heading = np.float64(wrist), None   # Gatling: wrist at the last pump, its direction
        self.pumps, self.gatling_until = [], -1.0


class Gomu:
    def __init__(self, frame_w, frame_h):
        self.w, self.h = frame_w, frame_h
        self.grips = []
        self.t = 0.0                    # seconds of play, for timers
        self.gear, self.gear_left = None, 0.0   # active gear (2 or 4) and its seconds left
        self.sign, self.sign_held, self.sign_missed = None, 0.0, 0   # gear sign being held

    def _reach(self, start, direction, palm, cap=PISTOL):
        """Fist travel along direction (at most cap palm sizes), stopped so the fist stays in the frame."""
        limits = [cap * palm]
        for p, d, size in zip(start, direction, (self.w, self.h)):
            if abs(d) > 1e-6:
                limits.append(((size - 1 - p) if d > 0 else p) / abs(d) - palm)
        return max(0.0, min(limits))

    def update(self, hands, dt):
        """hands: list of 21-landmark pixel lists. Returns (grab, tip, radius, back) stretches to draw."""
        dt = min(dt, MAX_DT)
        self.t += dt
        self.gear_left -= dt
        if self.gear_left <= 0:
            self.gear = None
        signs = []
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
            if kind in GEARS:                                 # a gear sign: not a grab
                signs.append(kind)
                kind = None
            if kind is None and grip.kind and grip.missed < GRACE:
                grip.missed += 1                              # likely a misread: hold the grab as it is
                continue
            grip.missed = 0
            if kind != grip.kind and kind:                    # a new grab: start from scratch
                grip.spring = Spring()
                grip.shape = kind
                if kind == "pinch":
                    grip.src = np.mean([lm[4], lm[8]], axis=0)  # image point under the fingers, fixed
            grip.kind = kind
            if kind == "fist":
                grip.src = np.mean([lm[0], lm[9]], axis=0)       # fist centre (fingers curled to the palm)
                grip.dir = np.subtract(lm[9], lm[0]) / palm
                grip.spring.step(grip.dir * self._reach(grip.src, grip.dir, palm), dt,
                                 GEAR2_BOOST if self.gear == 2 else 1.0)
                self._pump(grip, lm[0])
            elif kind == "pinch":
                grip.spring.hold(PULL_GAIN * (np.mean([lm[4], lm[8]], axis=0) - grip.src), dt)
        self._gear_sign(signs, dt)
        for grip in free:                                     # hand gone: let go after GRACE frames
            grip.missed += 1
            if grip.missed > GRACE:
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
            if fist:   # overshoot must not pull the fist back into the arm, nor push it off the frame
                edge = self._reach(grip.src, grip.dir, grip.palm, cap=math.inf)
                grip.spring.x = min(max(0.0, float(grip.spring.x @ grip.dir)), edge) * grip.dir
            radius = grip.palm * (0.9 if fist else 0.45)
            if fist and grip.kind == "fist" and self.t < grip.gatling_until:
                out += self._gatling(grip, radius)
            out.append((grip.src, grip.src + grip.spring.x, radius, radius * (2.0 if fist else 1.2)))
        self.grips = [g for g in self.grips if g.kind or g.shape or g not in free]
        return out

    def _gear_sign(self, signs, dt):
        """A gear sign held GEAR_HOLD s switches its gear on (ending the other); once per hold."""
        sign = signs[0] if signs else None
        if sign is None and self.sign and self.sign_missed < GRACE:
            self.sign_missed += 1                             # a misread frame doesn't break the hold
            return
        self.sign_missed = 0
        if sign != self.sign:
            self.sign, self.sign_held = sign, 0.0
        if sign is None:
            return
        before, self.sign_held = self.sign_held, self.sign_held + dt
        if before < GEAR_HOLD <= self.sign_held:
            self.gear, self.gear_left = GEARS[sign], GEAR_TIME

    def _pump(self, grip, wrist):
        """Count direction reversals of the fist; enough of them in a short window starts the Gatling."""
        move = np.subtract(wrist, grip.anchor)
        dist = np.hypot(*move)
        if dist < PUMP_DIST * grip.palm:
            return
        heading = move / dist
        if grip.heading is not None and heading @ grip.heading < -0.5:
            grip.pumps = [t for t in grip.pumps if t > self.t - GATLING_WINDOW] + [self.t]
            if len(grip.pumps) >= GATLING_PUMPS:
                grip.gatling_until = self.t + GATLING_HOLD
        grip.anchor, grip.heading = np.float64(wrist), heading

    def _gatling(self, grip, radius):
        """Extra fists fanned around the main one, each pumping in and out at its own phase.
        All share the main grab, so draw them from a clean copy of the frame (stretch's image)."""
        reach = max(float(np.hypot(*grip.spring.x)), 2 * grip.palm)
        out = []
        for i, angle in enumerate(np.linspace(-GATLING_SPREAD, GATLING_SPREAD, GATLING_FISTS)):
            c, s = math.cos(angle), math.sin(angle)
            d = np.array([c * grip.dir[0] - s * grip.dir[1], s * grip.dir[0] + c * grip.dir[1]])
            length = reach * (0.4 + 0.6 * abs(math.sin(self.t * 25 + i * 1.7)))
            out.append((grip.src, grip.src + d * length, radius * 0.8, radius * 1.6))
        return out

    def gatling(self):
        return any(g.kind == "fist" and self.t < g.gatling_until for g in self.grips)

    def pistol(self):
        """A fist is out far enough to call the attack."""
        return any(g.kind == "fist" and np.hypot(*g.spring.x) > 2 * g.palm for g in self.grips)


def gear_second(frame, mask, t):
    """Gear Second look, in place: the person (mask 1) tinted pink-red, pulsing like a heartbeat."""
    a = (np.clip(mask, 0, 1) * GEAR2_TINT * (0.8 + 0.2 * math.sin(t * 8)))[..., None]
    frame[:] = (frame * (1 - a) + np.float32((150, 90, 255)) * a).astype(np.uint8)


def steam(particles, mask, count=8):
    """Puffs of steam rising from random points on the person."""
    ys, xs = np.nonzero(mask > 0.5)
    for i in np.random.randint(0, len(xs), count) if len(xs) else []:
        particles.burst((xs[i], ys[i]), (70, 70, 80), count=3, speed=60.0, life=1.0)


def play(cap):
    """One Piece mode loop until Q ("quit") or M ("menu")."""
    detector = HandDetector(max_hands=2)
    gomu, last, was_pistol, was_gear = None, time.monotonic(), None, None
    segmenter, particles = None, ParticleSystem()   # the selfie mask is only made while a gear is on
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
        stretches = gomu.update([detector.get_landmark_list(lm, frame.shape) for lm in hands], dt)
        mask = None
        if gomu.gear:
            segmenter = segmenter or mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            mask = segmenter.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).segmentation_mask
        clean = frame.copy()
        for grab, tip, radius, back in stretches:
            stretch(frame, grab, tip, radius, back, image=clean)
        if gomu.gear == 2:
            gear_second(frame, mask, gomu.t)
            steam(particles, mask)
        particles.update(dt)
        particles.draw(frame)
        if gomu.gear and gomu.gear != was_gear:
            play_sound(150, 500)
        was_gear = gomu.gear
        if gomu.gear:
            text = f"GEAR {'SECOND' if gomu.gear == 2 else 'FOURTH'}  {math.ceil(gomu.gear_left)}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (180, 105, 255), 2)

        pistol = "GATLING" if gomu.gatling() else "PISTOL" if gomu.pistol() else None
        if pistol and pistol != was_pistol:
            play_sound(250, 700) if pistol == "PISTOL" else play_sound(400, 900)
        was_pistol = pistol
        if pistol:
            text = f"GOMU GOMU NO {pistol}!"
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.3, 3)[0]
            cv2.putText(frame, text, ((w - size[0]) // 2, 60), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 220), 3)
        cv2.putText(frame, "ONE PIECE", (10, h - 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, "Fist: Pistol (pump: Gatling) | Pinch, pull | V: Gear 2 | M, Q", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.imshow("Jutsu Code", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return "quit"
        if key == ord('m'):
            return "menu"
