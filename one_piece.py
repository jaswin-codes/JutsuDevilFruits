# one_piece.py - One Piece mode: Luffy's Gomu Gomu rubber body over the webcam image.
#   Fist:  Gomu Gomu no Pistol - the fist shoots out the way the hand points, the arm stretching
#          behind it; open the hand and it snaps back.
#          Pump the fist back and forth fast: Gomu Gomu no Gatling, a blur of fists.
#   V sign (index + middle up), held: Gear Second for GEAR_TIME s - pink steaming body, faster Pistol.
#   Four fingers up, thumb tucked, held: Gear Fourth - puffed-up arms, chest and huge fists.
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
PINCH_GRACE = 6          # a pinch near the face is often half hidden: it keeps its grab longer
PINCH_REACH = 1.15       # pinch: index tip at least this much farther from the wrist than its knuckle (not curled)
PUMP_DIST = 0.45         # Gatling: a pump counts once the wrist has travelled this many palm sizes (beats jitter)
GATLING_PUMPS, GATLING_WINDOW = 4, 1.0   # this many direction reversals within this many seconds
GATLING_HOLD = 0.5       # seconds the Gatling keeps going after the last pump
GATLING_FISTS = 6        # extra fists drawn around the main one
GATLING_SPREAD = 0.45    # radians either side of the fist direction
ARM_WIDTH, ARM_MAX = 0.5, 40     # Pistol arm half-width: this many palm sizes, but at most this many pixels
FACE_BASE, FACE_TIP = 0.75, 0.4  # pinch tube half-width in palm sizes: wide on the face, narrow at the fingers
EDGE = 2.0               # stretch: soft edge, in pixels
SHADE = 0.45             # stretch: how much darker the tube's sides are than its middle
OUTLINE, OUTLINE_DARK = 1.5, 0.45  # stretch: outline width in pixels and its brightness
GEAR_HOLD = 0.5          # seconds a gear sign must be held to switch the gear on
GEAR_TIME = 10.0         # seconds a gear lasts
GEAR2_BOOST = 4.0        # Gear Second spring stiffness factor: Pistol is sqrt(4) = 2x faster
GEAR2_TINT = 0.35        # how pink the body gets (0..1)
THUMB_IN = 0.5           # "four" sign: thumb tip within this many palm sizes of the ring knuckle
GEAR4_PUFF = 0.45        # Gear Fourth bulge: the centre is magnified 1 / (1 - 0.45) = 1.8x
GEAR4_FIST = 1.6         # Gear Fourth fists are this much bigger


def stretch(frame, grab, tip, radius, back=None, image=None, width=None):
    """Rubber stretch, in place. The disc of `radius` around `grab` (the fist, the pinched skin)
    is carried whole to `tip`, crisp; the strip from `back` pixels behind grab up to that disc is
    drawn stretched over the rest of the gap, as a shaded, outlined tube whose half-width goes
    from width[0] at its start to width[1] at the tip end (default radius). Nothing behind the
    strip moves, so there is no fold. image: what to read from (default frame), so several
    stretches of one grab copy the clean fist."""
    G, T = np.float32(grab), np.float32(tip)
    gap = float(np.hypot(*(T - G)))
    if gap < 1:
        return
    back = radius if back is None else back
    base, neck = (radius, radius) if width is None else width
    c = min(radius, 0.75 * back)              # carried disc; the strip behind it keeps 1/4 of back at least
    u = (T - G) / gap
    n = np.float32([-u[1], u[0]])
    A = G - back * u                          # anchor: the start of the stretched strip
    length = back + gap                       # anchor to tip on screen
    reach = max(radius, base, neck)
    h, w = frame.shape[:2]
    x0, y0 = np.maximum(np.floor(np.minimum(A, T) - reach), 0).astype(int)
    x1, y1 = np.minimum(np.ceil(np.maximum(A, T) + reach) + 1, (w, h)).astype(int)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    qx, qy = xs - A[0], ys - A[1]
    a, b = qx * u[0] + qy * u[1], qx * n[0] + qy * n[1]      # along / across the band
    src = np.where(a <= 0, a, np.where(a < length - c, a * ((back - c) / (length - c)), a - gap))
    patch = cv2.remap(frame if image is None else image, A[0] + src * u[0] + b * n[0],
                      A[1] + src * u[1] + b * n[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    half = base + (neck - base) * np.clip(a / length, 0, 1)   # tube half-width here
    edge = half - np.abs(b)                                   # pixels inside the tube's side
    tube = np.where((a >= 0) & (a <= length), np.clip(edge / EDGE, 0, 1), 0)
    disc = np.clip((radius - np.hypot(a - length, b)) / EDGE, 0, 1)
    # tube look: darker towards the sides (round, not flat) and a dark outline; fades in from the anchor
    look = (1 - SHADE * (b / half) ** 2) * np.where(edge < EDGE + OUTLINE, OUTLINE_DARK, 1.0)
    look = 1 - (1 - look) * np.clip(a / base, 0, 1) * (1 - disc) * (a < length - c)
    alpha = np.maximum(tube, disc)[..., None]
    roi = frame[y0:y1, x0:x1]
    roi[:] = np.clip(patch * look[..., None] * alpha + roi * (1 - alpha), 0, 255).astype(np.uint8)


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


def fingers_out(lm):
    """[index, middle, ring, pinky] extended (tip farther from the wrist than its middle joint)."""
    return [_d(lm[t], lm[0]) > _d(lm[p], lm[0]) for t, p in ((8, 6), (12, 10), (16, 14), (20, 18))]


def gesture(lm, pinching=False):
    """'fist', 'pinch', 'two' (V sign), 'four' (thumb tucked) or None from 21 hand landmarks in pixels. pinching: last
    frame's answer was 'pinch' (looser release threshold, so a grab doesn't flicker)."""
    out = fingers_out(lm)
    palm = max(_d(lm[0], lm[9]), 1.0)
    # Pinch first, whatever the other fingers do (they often curl and hide the thumb near the face).
    # A fist is told apart by its index tip curled back into the palm, not reaching past its knuckle.
    index_reaches = _d(lm[8], lm[0]) > PINCH_REACH * _d(lm[5], lm[0])
    if index_reaches and _d(lm[4], lm[8]) < (PINCH_OFF if pinching else PINCH_ON) * palm:
        return "pinch"
    if not any(out):
        return "fist"
    if out == [True, True, False, False]:
        return "two"
    if all(out) and _d(lm[4], lm[13]) < THUMB_IN * palm:
        return "four"
    return None


GEARS = {"two": 2, "four": 4}   # gear sign -> gear


def _grace(grip):
    return PINCH_GRACE if grip.kind == "pinch" else GRACE


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
        """hands: list of 21-landmark pixel lists. Returns (grab, tip, radius, back, width) stretches to draw."""
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
            misread = kind is None or (grip.kind == "pinch" and kind == "fist")   # hidden fingers read as a fist
            if misread and grip.kind and grip.missed < _grace(grip):
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
        for grip in free:                                     # hand gone: let go after a few frames
            grip.missed += 1
            if grip.missed > _grace(grip):
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
            big = GEAR4_FIST if fist and self.gear == 4 else 1
            if fist:
                radius, arm = 0.9 * grip.palm * big, min(ARM_WIDTH * grip.palm, ARM_MAX) * big
                width, back = (arm, arm), 2.0 * radius
            else:
                radius, width = FACE_TIP * grip.palm, (FACE_BASE * grip.palm, FACE_TIP * grip.palm)
                back = 1.6 * width[0]                             # anchor deep in the face
            if fist and grip.kind == "fist" and self.t < grip.gatling_until:
                out += self._gatling(grip, radius, arm)
            out.append((grip.src, grip.src + grip.spring.x, radius, back, width))
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

    def _gatling(self, grip, radius, arm):
        """Extra fists fanned around the main one, each pumping in and out at its own phase.
        All share the main grab, so draw them from a clean copy of the frame (stretch's image)."""
        reach = max(float(np.hypot(*grip.spring.x)), 2 * grip.palm)
        out = []
        for i, angle in enumerate(np.linspace(-GATLING_SPREAD, GATLING_SPREAD, GATLING_FISTS)):
            c, s = math.cos(angle), math.sin(angle)
            d = np.array([c * grip.dir[0] - s * grip.dir[1], s * grip.dir[0] + c * grip.dir[1]])
            length = reach * (0.4 + 0.6 * abs(math.sin(self.t * 25 + i * 1.7)))
            out.append((grip.src, grip.src + d * length, radius * 0.8, radius * 1.6, (arm * 0.8, arm * 0.8)))
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


def bulge(frame, centre, radius, strength=GEAR4_PUFF):
    """Puff up a disc, in place: the middle is magnified, the rim stays put (no seam, no fold)."""
    c = np.float32(centre)
    h, w = frame.shape[:2]
    x0, y0 = np.maximum(np.floor(c - radius), 0).astype(int)
    x1, y1 = np.minimum(np.ceil(c + radius) + 1, (w, h)).astype(int)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xs - c[0], ys - c[1]
    r2 = (dx * dx + dy * dy) / radius ** 2
    scale = np.where(r2 < 1, 1 - strength * (1 - r2), 1)   # source radius r(1 - s + s r^2/R^2) grows with r
    patch = cv2.remap(frame, c[0] + dx * scale, c[1] + dy * scale, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
    roi = frame[y0:y1, x0:x1]
    inside = r2 < 1
    roi[inside] = patch[inside]


def puffs(hands, mask=None):
    """Gear Fourth bulges as (centre, radius): each hand and its forearm, and the chest (middle of
    the selfie mask) when there is one."""
    out = []
    for lm in hands:
        wrist, knuckle = np.float32(lm[0]), np.float32(lm[9])
        palm = max(float(np.hypot(*(knuckle - wrist))), 1.0)
        out += [((wrist + knuckle) / 2, 1.8 * palm), (wrist - (knuckle - wrist) * 1.3, 1.6 * palm)]
    if mask is not None:
        ys, xs = np.nonzero(mask > 0.5)
        if len(xs) > 100:
            out.append(((xs.mean(), ys.mean() + 0.15 * np.ptp(ys)), 0.6 * math.sqrt(len(xs) / math.pi)))
    return out


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
        hands = [detector.get_landmark_list(lm, frame.shape) for lm in hands]
        stretches = gomu.update(hands, dt)
        mask = None
        if gomu.gear:
            segmenter = segmenter or mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            mask = segmenter.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).segmentation_mask
        if gomu.gear == 4:
            for centre, radius in puffs(hands, mask):
                bulge(frame, centre, radius)
        clean = frame.copy()
        for grab, tip, radius, back, width in stretches:
            stretch(frame, grab, tip, radius, back, clean, width)
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
            cv2.putText(frame, text, (160, h - 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (180, 105, 255), 2)

        pistol = "GATLING" if gomu.gatling() else "PISTOL" if gomu.pistol() else None
        if pistol and pistol != was_pistol:
            play_sound(250, 700) if pistol == "PISTOL" else play_sound(400, 900)
        was_pistol = pistol
        if pistol:
            text = f"GOMU GOMU NO {pistol}!"
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.3, 3)[0]
            cv2.putText(frame, text, ((w - size[0]) // 2, 60), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 220), 3)
        cv2.putText(frame, "ONE PIECE", (10, h - 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, "Fist: Pistol (pump: Gatling) | Pinch, pull | V: Gear 2 | 4: Gear 4 | M, Q", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.imshow("JutsuDevilFruits", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return "quit"
        if key == ord('m'):
            return "menu"
