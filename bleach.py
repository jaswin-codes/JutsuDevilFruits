# bleach.py - Bleach world: draw a soul-reaper sword out of thin air, then transform.
#   Fist held low, pulled up fast: a katana forms in the fist and follows its angle.
#   Swing it fast: Getsuga Tensho, a crescent slash that flies off the blade.
#   Grip the sword wrist with your other hand for 1 s: Bankai (dark room, red aura, black blade).
#   Drag an open hand down over your face: a Hollow mask (bigger slashes while it lasts).
# Shinigami (the logic) and the draw functions are pure (no camera) and tested; play() wires them up.
import collections
import math
import time
import cv2
import mediapipe as mp
import numpy as np
from anime_effects import _add_light
from battle import Battle
from effects import ParticleSystem, play_sound
from hand_detector import HandDetector
from mera import fist_centre
from one_piece import MAX_DT, fingers_out, gesture

# Sword and swing thresholds tuned on a webcam (a swing blurs the hand: tracking drops frames at ~10-15 fps)
DRAW_LOW = 0.55       # draw: the fist starts in the lower part of the frame (fraction of its height)...
DRAW_RISE = 1.5       # ...and rises this many palm sizes...
DRAW_WINDOW = 0.4     # ...within this many seconds
FORM_TIME = 0.3       # the blade grows out of the fist over this long
SWING_LOCK = 0.5      # no Getsuga this soon after drawing (the draw itself is a fast move)
BLADE = 4.5           # blade length in palm sizes (wrist to middle knuckle: doesn't shrink as the fist turns)
EDGE_ON = 0.35        # knuckle row shorter than this many palm sizes: the fist points at the camera, keep the angle
ANGLE_SMOOTH = 0.35   # blade angle: weight of the new reading each frame (the blade's length magnifies jitter)
SWING_DIST = 3.0      # Getsuga: the blade tip travels this many palm sizes...
SWING_PALM = 90.0     # ...counting a palm as at most this many px (close to the camera the frame is too small otherwise)...
SWING_WINDOW = 0.3    # ...within this many seconds
SWING_COOLDOWN = 0.35 # seconds between slashes: one swing's follow-through is not a second swing
GETSUGA_TIME = 1.0    # seconds a slash flies
GETSUGA_SPEED = 900.0 # px/s
SHEATHE = 1.5         # seconds without the sword fist and the sword is gone
STICK = 4.0           # the sword stays in the fist nearest the last one, within this many palm sizes
                      # (a fast swing at a low frame rate moves the fist far in one frame)
STICK_GROW = 5.0      # that reach grows by this fraction per second the fist is unseen (it kept moving)
HOLD_OPEN = 3         # a hand holding the sword only lets go with this many fingers out (a turned fist reads as loose)
GRIP = 1.3            # Bankai: the other hand's palm within this many palm sizes of the sword wrist...
BANKAI_HOLD = 1.0     # ...for this long
GRIP_GRACE = 0.2      # overlapping hands drop out of tracking: a gap this short keeps the grip timer
SWIPE_TIME = 0.6      # mask: an open hand goes from the forehead to below the chin within this long
FACE_AGE = 1.0        # a face seen this recently still counts (the swiping hand hides the face)
MASK_TIME = 15.0      # seconds the Hollow mask lasts
STEP_DIST = 0.15      # Flash Step: the body's centre moves this fraction of the frame width...
STEP_WINDOW = 0.25    # ...within this many seconds
STEP_COOLDOWN = 1.0   # seconds between Flash Steps
FLASH_TIME = 0.5      # seconds the afterimages take to fade
GHOST_FRAMES = 6      # recent frames kept to leave behind as afterimages
DODGE = 0.6           # Hollow hunt: the next hit within this long of a Flash Step misses you
SCATTER_HOLD = 0.25   # Bankai: the sword hand stays open this long and the blade scatters (a flicker doesn't)
PETALS = 350          # Senbonzakura: petals in the swarm
SWARM_RADIUS = 70.0   # px: Hollows this close to the swarm's centre are cut
SWARM_POWER = 1.5     # the swarm's damage against the blade's
GETSUGA_DAMAGE = 100.0  # Hollow hunt: one sealed slash kills a wave-1 Hollow; Bankai doubles it

STEEL, BLUE, RED, PINK = (205, 200, 195), (255, 215, 130), (40, 30, 235), (190, 130, 255)   # BGR
OVAL = sorted({i for pair in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in pair})


class Shinigami:
    def __init__(self):
        self.t = 0.0
        self.sword_at = None        # when the sword was drawn (None: no sword)
        self.fist = self.hilt = self.angle = None
        self.palm, self.seen = 60.0, 0.0   # sword hand's palm size; last time its fist was seen
        self.rises = []             # (t, centre) of the drawing fist over the last DRAW_WINDOW s
        self.tips = []              # (t, blade tip) over the last SWING_WINDOW s
        self.grip_since = self.grip_seen = None
        self.bankai = self.scattered = False   # scattered: the blade is a swarm of petals
        self.open_since, self.swarm = None, None   # when the sword hand opened; where the petals are sent
        self.slashes = []           # (t fired, origin, unit direction, size, bankai)
        self.face, self.face_at = None, -math.inf   # last face box (top, bottom, left, right)
        self.bodies, self.flash_at = [], -math.inf   # (t, body centre) over the last STEP_WINDOW s; last Flash Step
        self.swipe = None           # time an open hand was last on the forehead
        self.mask_at = None

    def update(self, hands, dt, height, face=None, body=None):
        """hands: 21-landmark pixel lists; face: (top, bottom, left, right) if seen this frame; body: centre of the person as a fraction of the width.
        Returns this frame's events: 'draw', 'getsuga', 'bankai', 'mask', 'sheathe', 'scatter', 'recall', 'flash'."""
        self.t += min(dt, MAX_DT)
        events = []
        if face is not None:
            self.face, self.face_at = face, self.t
        self.slashes = [s for s in self.slashes if self.t - s[0] < GETSUGA_TIME]
        palms = [max(math.dist(lm[0], lm[9]), 1.0) for lm in hands]
        fists = [i for i, lm in enumerate(hands) if gesture(lm) == "fist"]
        if self.sword_at is None:
            self._draw(hands, fists, palms, height, events)
        else:
            self._hold(hands, [i for i, lm in enumerate(hands) if sum(fingers_out(lm)) < HOLD_OPEN], palms, events)
        self._mask(hands, events)
        self._step(body, events)
        return events

    def masked(self):
        return self.mask_at is not None and self.t - self.mask_at < MASK_TIME

    def tip(self):
        grown = min(1.0, (self.t - self.sword_at) / FORM_TIME)
        return self.hilt + BLADE * self.palm * grown * np.array([math.cos(self.angle), math.sin(self.angle)])

    def _aim(self, lm):
        """Blade along the knuckle row, pinky side to index side (how a sword sits in a fist)."""
        row = np.subtract(lm[5], lm[17])
        if np.hypot(*row) >= EDGE_ON * self.palm or self.angle is None:
            new = math.atan2(row[1], row[0])
            if self.angle is None:
                self.angle = new
            else:   # step part of the way, the short way round
                self.angle += ANGLE_SMOOTH * ((new - self.angle + math.pi) % (2 * math.pi) - math.pi)
        self.hilt = self.fist + 0.5 * self.palm * np.array([math.cos(self.angle), math.sin(self.angle)])

    def _draw(self, hands, fists, palms, height, events):
        if not fists:
            return            # a dropped frame keeps the samples; they age out on their own
        last = self.rises[-1][1] if self.rises else None
        i = (min(fists, key=lambda i: math.dist(fist_centre(hands[i]), last)) if last is not None
             else max(fists, key=lambda i: fist_centre(hands[i])[1]))
        centre = fist_centre(hands[i])
        if last is not None and math.dist(centre, last) > STICK * palms[i]:
            self.rises = []   # a different fist
        self.rises = [s for s in self.rises if s[0] >= self.t - DRAW_WINDOW] + [(self.t, centre)]
        low = max(y for _, (_, y) in self.rises)
        if low > DRAW_LOW * height and low - centre[1] > DRAW_RISE * palms[i]:
            self.sword_at, self.seen, self.fist, self.palm, self.angle = self.t, self.t, centre, palms[i], None
            self._aim(hands[i])
            self.rises, self.tips = [], []
            events.append("draw")

    def _step(self, body, events):
        if body is None:              # nobody in frame: don't compare across the gap
            self.bodies = []
            return
        self.bodies = [s for s in self.bodies if s[0] >= self.t - STEP_WINDOW] + [(self.t, body)]
        if abs(body - self.bodies[0][1]) > STEP_DIST and self.t - self.flash_at > STEP_COOLDOWN:
            self.flash_at, self.bodies = self.t, []
            events.append("flash")

    def _sheathe(self, events):
        self.sword_at, self.bankai, self.scattered, self.grip_since, self.grip_seen = None, False, False, None, None
        events.append("sheathe")

    def _scatter(self, hands, held, palms, reach, events):
        """Bankai only: holding the sword hand open turns the blade into petals, sent at the other hand's fingertip
        (or kept around the open hand); a real fist calls the blade back. True while the blade is petals."""
        close = [i for i in range(len(hands)) if math.dist(fist_centre(hands[i]), self.fist) < reach * palms[i]]
        if not self.bankai:
            return False
        if not self.scattered and (not close or set(close) & set(held)):
            self.open_since = None
            return False
        if not close:                 # scattered and the hand is lost
            if self.t - self.seen > SHEATHE:
                self._sheathe(events)
            return True
        i = min(close, key=lambda i: math.dist(fist_centre(hands[i]), self.fist))
        self.fist, self.palm, self.seen = fist_centre(hands[i]), palms[i], self.t
        if not self.scattered:
            self.open_since = self.open_since if self.open_since is not None else self.t
            if self.t - self.open_since < SCATTER_HOLD:
                return False
            self.scattered, self.open_since = True, None
            events.append("scatter")
        elif gesture(hands[i]) == "fist":
            self.scattered, self.tips = False, []
            events.append("recall")
            return False
        others = [lm for j, lm in enumerate(hands) if j != i]
        self.swarm = np.float32(others[0][8] if others else self.fist)
        return True

    def _hold(self, hands, fists, palms, events):
        reach = STICK * (1 + STICK_GROW * (self.t - self.seen))
        if self._scatter(hands, fists, palms, reach, events):
            return
        near = [i for i in fists if math.dist(fist_centre(hands[i]), self.fist) < reach * palms[i]]
        if not near:
            if self.t - self.seen > SHEATHE:
                self._sheathe(events)
            return
        # The sword stays in the fist nearest the last one, even if the gripping hand is a fist too
        i = min(near, key=lambda i: math.dist(fist_centre(hands[i]), self.fist))
        self.fist, self.palm, self.seen = fist_centre(hands[i]), palms[i], self.t
        self._aim(hands[i])
        tip = self.tip()
        self.tips = [s for s in self.tips if s[0] >= self.t - SWING_WINDOW] + [(self.t, tip)]
        swing = tip - self.tips[0][1]
        rested = not self.slashes or self.t - self.slashes[-1][0] > SWING_COOLDOWN
        if self.t - self.sword_at > SWING_LOCK and rested and np.hypot(*swing) > SWING_DIST * min(self.palm, SWING_PALM):
            size = self.palm * (2.0 if self.bankai else 1.6) * (1.5 if self.masked() else 1.0)
            self.slashes.append((self.t, tip, swing / np.hypot(*swing), size, self.bankai, set()))   # set: Hollows cut
            self.tips = [(self.t, tip)]   # one swing, one slash
            events.append("getsuga")
        gripping = any(math.dist(fist_centre(hands[j]), hands[i][0]) < GRIP * self.palm
                       for j in range(len(hands)) if j != i)
        if gripping:
            self.grip_since = self.grip_since if self.grip_since is not None else self.t
            self.grip_seen = self.t
            if not self.bankai and self.t - self.grip_since >= BANKAI_HOLD:
                self.bankai = True
                events.append("bankai")
        elif self.grip_seen is None or self.t - self.grip_seen > GRIP_GRACE:
            self.grip_since = None

    def _mask(self, hands, events):
        if self.mask_at is not None and not self.masked():
            self.mask_at = None
        if self.t - self.face_at > FACE_AGE or self.masked():
            self.swipe = None
            return
        top, bottom, left, right = self.face
        for lm in hands:
            x, y = fist_centre(lm)
            if not all(fingers_out(lm)) or not left <= x <= right:
                continue
            if y < top + 0.35 * (bottom - top):
                self.swipe = self.t          # open palm on the forehead
            elif y > bottom and self.swipe is not None and self.t - self.swipe < SWIPE_TIME:
                self.mask_at, self.swipe = self.t, None
                events.append("mask")


def _unit(v):
    v = np.float32(v)
    return v / max(float(np.hypot(*v)), 1e-6)


def draw_sword(frame, hilt, tip, bankai=False):
    """A katana from the guard (hilt) to the tip, with a glow of spirit energy: blue sealed, red in Bankai."""
    hilt, tip = np.float32(hilt), np.float32(tip)
    length = float(np.hypot(*(tip - hilt)))
    if length < 2:
        return
    u = _unit(tip - hilt)
    n = np.float32((-u[1], u[0]))
    w = max(3.0, 0.045 * length)            # half width at the guard
    glow = RED if bankai else BLUE
    centre = (hilt + tip) / 2

    def light(layer, c):
        off = np.float32(c) - centre
        cv2.line(layer, tuple(np.int32(hilt + off)), tuple(np.int32(tip + off)), glow, int(2.5 * w), cv2.LINE_AA)
    _add_light(frame, centre, int(length / 2 + 4 * w), light, bloom=1)
    blade = np.int32([hilt + n * w, hilt + 0.88 * (tip - hilt) + n * 0.8 * w, tip,
                      hilt + 0.84 * (tip - hilt) - n * 0.5 * w, hilt - n * 0.6 * w])
    cv2.fillPoly(frame, [blade], (25, 20, 20) if bankai else STEEL, cv2.LINE_AA)
    cv2.line(frame, tuple(np.int32(hilt - n * 0.4 * w)), tuple(np.int32(tip)),   # the cutting edge
             RED if bankai else (255, 255, 255), max(1, int(w / 3)), cv2.LINE_AA)
    cv2.line(frame, tuple(np.int32(hilt - n * 2.4 * w)), tuple(np.int32(hilt + n * 2.4 * w)),   # guard
             (20, 20, 20) if bankai else (40, 150, 200), max(2, int(1.2 * w)), cv2.LINE_AA)


def crescent(centre, direction, radius):
    """Polygon of a crescent facing `direction`: the part of a circle in front of the same circle
    moved back by 0.45 radius. Thick in the middle, pointed at both ends."""
    u = _unit(direction)
    base, back = math.atan2(u[1], u[0]), 0.45 * radius
    outer_end, inner_end = math.acos(-back / (2 * radius)), math.acos(back / (2 * radius))
    outer = [np.float32(centre) + radius * np.float32((math.cos(base + a), math.sin(base + a)))
             for a in np.linspace(-outer_end, outer_end, 28)]
    inner = [np.float32(centre) - back * u + radius * np.float32((math.cos(base + a), math.sin(base + a)))
             for a in np.linspace(inner_end, -inner_end, 20)]
    return np.float32(outer + inner)


class Petals:
    """Senbonzakura: a swarm that peels off the blade, then chases and circles a target."""
    def __init__(self, hilt, tip, rng=None):
        self.rng = rng or np.random.default_rng()
        along = self.rng.random((PETALS, 1))
        self.pos = np.float32(hilt) + along * (np.float32(tip) - np.float32(hilt)) + self.rng.normal(0, 8, (PETALS, 2))
        self.vel = self.rng.normal(0, 120, (PETALS, 2))
        self.shade = self.rng.random(PETALS)

    def update(self, dt, target):
        dt = min(dt, MAX_DT)
        to = np.float32(target) - self.pos
        swirl = np.stack([-to[:, 1], to[:, 0]], axis=1)
        self.vel = (self.vel + (14 * to + 5 * swirl + self.rng.normal(0, 500, (PETALS, 2))) * dt) * math.exp(-4 * dt)
        self.pos += self.vel * dt

    def centre(self):
        return self.pos.mean(axis=0)


def draw_petals(frame, petals):
    for (x, y), shade in zip(petals.pos.astype(int).tolist(), petals.shade.tolist()):
        cv2.circle(frame, (x, y), 2, (150 + 70 * shade, 90 + 110 * shade, 255), -1, cv2.LINE_AA)


def getsuga_at(origin, direction, age, size, duration=GETSUGA_TIME):
    """Where a slash is after flying for age seconds: its centre and radius."""
    return np.float32(origin) + _unit(direction) * GETSUGA_SPEED * age, size * (1 + 1.5 * age / duration)


def draw_getsuga(frame, origin, direction, age, size, bankai=False, duration=GETSUGA_TIME):
    """Getsuga Tensho: a crescent of energy flying off along direction, growing and fading.
    Sealed it is blue-white light; in Bankai a black core with a red edge."""
    if not 0 <= age < duration:
        return
    fade = 1 - age / duration
    centre, radius = getsuga_at(origin, direction, age, size, duration)
    shape = crescent(centre, direction, radius)
    x, y, bw, bh = cv2.boundingRect(np.int32(shape))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + bw, frame.shape[1]), min(y + bh, frame.shape[0])
    if bankai and x1 > x0 and y1 > y0:   # the black core darkens what's behind it (only its own box: cheap)
        roi = frame[y0:y1, x0:x1]
        core = np.zeros(roi.shape[:2], np.uint8)
        cv2.fillPoly(core, [np.int32(shape - (x0, y0))], 255, cv2.LINE_AA)
        a = (core.astype(np.float32) / 255 * 0.85 * fade)[..., None]
        roi[:] = (roi * (1 - a) + np.float32((15, 5, 10)) * a).astype(np.uint8)
    colour = tuple(int(c * fade) for c in (RED if bankai else (255, 235, 170)))

    def light(layer, c):
        pts = np.int32(shape - centre + np.float32(c))
        if bankai:
            cv2.polylines(layer, [pts], True, colour, max(2, int(radius / 10)), cv2.LINE_AA)
        else:
            cv2.fillPoly(layer, [pts], colour, cv2.LINE_AA)
    _add_light(frame, centre, int(radius * 1.6), light)


def draw_bankai_aura(frame, mask, t):
    """Bankai look, in place: the room goes dark and red spirit flames rise off the person (mask 1).
    Flames are made at 1/4 size and everything stays uint8, to stay cheap."""
    h, w = frame.shape[:2]
    small = (cv2.resize(mask, (w // 4, h // 4)) > 0.5).astype(np.uint8)
    ring = cv2.dilate(small, np.ones((5, 5), np.uint8)) - small
    rng = np.random.default_rng(int(t * 24))          # a new flicker 24 times a second
    flames = ring.astype(np.float32) * (rng.random(ring.shape) < 0.35)   # sparse tongues, not a neat outline
    for _ in range(10):                                # each tongue licks upward, thinning out
        flames = np.maximum(flames, np.roll(flames, -1, axis=0) * 0.88)
    pulse = 0.85 + 0.15 * math.sin(t * 6)
    glow = np.clip(cv2.GaussianBlur(flames, (0, 0), 1.5)[..., None] * np.float32(RED) * 1.6 * pulse, 0, 255)
    np.copyto(frame, frame >> 1, where=(cv2.resize(mask, (w, h)) < 0.5)[..., None])   # dim all but the person
    cv2.add(frame, cv2.resize(glow.astype(np.uint8), (w, h)), dst=frame)


def face_geometry(points):
    """Face mesh pixel points (478, refined) -> (face outline polygon, the two iris centres left to right, mouth)."""
    points = np.float32(points)
    oval = cv2.convexHull(points[OVAL]).reshape(-1, 2)
    irises = sorted([points[468], points[473]], key=lambda p: p[0])
    return oval, irises, (points[13] + points[14]) / 2


def draw_hollow_mask(frame, oval, irises, mouth, fade=1.0):
    """A white bone mask over the face outline: red stripes down one side, black eyes with yellow
    irises, and a row of teeth over the mouth. The face's own shading shows through for depth."""
    x, y, bw, bh = cv2.boundingRect(np.int32(oval))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + bw, frame.shape[1]), min(y + bh, frame.shape[0])
    if x1 <= x0 or y1 <= y0 or fade <= 0:
        return
    off = np.float32((x0, y0))
    roi = frame[y0:y1, x0:x1]
    gap = max(float(np.hypot(*(irises[1] - irises[0]))), 1.0)
    u = _unit(irises[1] - irises[0])                       # across the face, left to right
    v = np.float32((-u[1], u[0])) * (1 if u[0] >= 0 else -1)   # down the face
    mid = (irises[0] + irises[1]) / 2 - off

    shade = cv2.cvtColor(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.float32)
    layer = np.clip(np.float32((225, 232, 238)) * 0.75 + shade * 0.35, 0, 255).astype(np.uint8)
    for side in (-0.85, -0.5):                             # red stripes down the left side
        top, bottom = mid + side * gap * u - 1.4 * gap * v, mid + (side - 0.12) * gap * u + 1.7 * gap * v
        cv2.line(layer, tuple(np.int32(top)), tuple(np.int32(bottom)), (30, 25, 190), max(2, int(0.14 * gap)),
                 cv2.LINE_AA)
    angle = math.degrees(math.atan2(u[1], u[0]))
    for iris in irises:
        c = tuple(np.int32(iris - off))
        cv2.ellipse(layer, c, (int(0.32 * gap), int(0.17 * gap)), angle, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)
        cv2.circle(layer, c, max(2, int(0.09 * gap)), (0, 210, 255), -1, cv2.LINE_AA)
        cv2.circle(layer, c, max(1, int(0.035 * gap)), (0, 0, 0), -1, cv2.LINE_AA)
    m = np.float32(mouth) - off
    cv2.line(layer, tuple(np.int32(m - 0.45 * gap * u)), tuple(np.int32(m + 0.45 * gap * u)), (60, 60, 60), 2,
             cv2.LINE_AA)
    for s in np.linspace(-0.4, 0.4, 7):                    # teeth
        cv2.line(layer, tuple(np.int32(m + s * gap * u - 0.13 * gap * v)),
                 tuple(np.int32(m + s * gap * u + 0.13 * gap * v)), (60, 60, 60), 2, cv2.LINE_AA)
    inside = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(inside, [np.int32(oval - off)], 255, cv2.LINE_AA)
    a = (inside.astype(np.float32) / 255 * 0.95 * fade)[..., None]
    roi[:] = (roi * (1 - a) + layer * a).astype(np.uint8)


def body_x(mask):
    """Centre of the person in a segmentation mask as a fraction of the width; None if hardly anyone is there."""
    column = (mask > 0.5).sum(axis=0)
    if column.sum() < 0.03 * mask.size:
        return None
    return float((column * np.arange(len(column))).sum() / column.sum() / len(column))


def merge_ghosts(ghosts):
    """Flash Step: the frames just before the lunge as one blue layer (colour, opacity), newest on top and strongest.
    Built once per step: blending every ghost on every frame took 100 ms."""
    layer = np.zeros_like(ghosts[0][0], np.float32)
    weight = np.zeros(ghosts[0][1].shape, np.float32)
    for k, (image, ghost) in enumerate(ghosts):
        person = ghost > 0.5
        layer[person] = 0.6 * image[person] + 0.4 * np.float32(BLUE)
        weight[person] = 0.2 + 0.4 * (k + 1) / len(ghosts)
    return layer, weight


def draw_afterimages(frame, layer, weight, mask, fade):
    """The merged afterimages over the room, the real person kept in front."""
    out = frame.astype(np.float32)
    out += (layer - out) * (weight * np.float32(fade))[:, :, None]
    frame[:] = np.where((mask > 0.5)[:, :, None], frame, out.astype(np.uint8))


def hunt(battle, soul, dt, petals=None):
    """The sword and its flying slashes against the Hollows. Returns the positions of those killed."""
    killed = []
    if petals is not None:
        killed += battle.melee(petals.centre(), SWARM_RADIUS, dt, SWARM_POWER)
    elif soul.sword_at is not None:
        killed += battle.melee(soul.tip(), 0.5 * soul.palm, dt)
    for fired, origin, direction, size, bankai, hit in soul.slashes:
        centre, radius = getsuga_at(origin, direction, soul.t - fired, size)
        killed += battle.slash(centre, radius, GETSUGA_DAMAGE * (2 if bankai else 1), hit)
    return killed


def draw_hollow(frame, m, r, t):
    """A Hollow: black body with a hole through the chest and a bone-white mask, bobbing as it walks."""
    x, y = int(m.x), int(m.y + 0.08 * r * math.sin(t * 7 + m.x * 0.05))
    r = int(r)
    bone = (255, 255, 255) if t - m.hit_time < 0.12 else (225, 235, 240)
    cv2.ellipse(frame, (x, y + r // 3), (r, int(1.1 * r)), 0, 0, 360, (35, 25, 30), -1, cv2.LINE_AA)
    cv2.circle(frame, (x, y + r // 2), r // 3, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, (x, y + r // 2), r // 3, RED, 2, cv2.LINE_AA)
    cv2.ellipse(frame, (x, y - r // 2), (int(0.7 * r), int(0.6 * r)), 0, 0, 360, bone, -1, cv2.LINE_AA)
    for ex in (-1, 1):
        cv2.ellipse(frame, (x + ex * r // 3, y - int(0.6 * r)), (r // 5, r // 8), 0, 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.line(frame, (x - r // 3, y - r // 5), (x + r // 3, y - r // 5), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(frame, (x, y - r), (x + r // 6, y - r // 2), RED, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (x - r, y - r - 12), (x + r, y - r - 6), (40, 40, 40), -1)
    cv2.rectangle(frame, (x - r, y - r - 12), (x - r + int(2 * r * max(m.hp, 0) / m.max_hp), y - r - 6), RED, -1)


def banner(frame, text, colour):
    w = frame.shape[1]
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.4, 3)[0]
    cv2.putText(frame, text, ((w - size[0]) // 2, 60), cv2.FONT_HERSHEY_DUPLEX, 1.4, colour, 3)


EVENTS = {   # event -> (sound sweep, banner text or None, banner colour)
    "draw": ((300, 1400), None, None),
    "getsuga": ((900, 200), "GETSUGA TENSHO!", BLUE),
    "bankai": ((90, 400), "BANKAI!", RED),
    "mask": ((160, 70), "HOLLOW MASK", (230, 230, 230)),
    "scatter": ((1300, 500), "SENBONZAKURA", PINK),
    "recall": ((500, 1300), None, None),
    "flash": ((1800, 1000), "FLASH STEP", BLUE),
}


def play(cap):
    """Bleach mode loop until Q ("quit") or M ("menu")."""
    detector = HandDetector(max_hands=2, detection_confidence=0.6, tracking_confidence=0.5)   # keep fast hands
    soul, particles, last = Shinigami(), ParticleSystem(), time.monotonic()
    segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)   # ~5 ms: Flash Step, Bankai
    face_mesh = geometry = None               # built only when a hand is near the face
    history, ghosts, flashed = collections.deque(maxlen=GHOST_FRAMES), None, -math.inf
    shown, shown_until = None, 0.0
    battle = None                             # the Hollow hunt, while H has it on
    petals = None                             # Senbonzakura swarm, while the blade is scattered
    while True:
        success, frame = cap.read()
        if not success:
            print("Camera stopped sending frames.")
            return "quit"
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        _, found = detector.find_hands(frame, draw=False)
        now = time.monotonic()
        dt, last = now - last, now
        hands = [detector.get_landmark_list(lm, frame.shape) for lm in found]
        face = None
        # Face mesh is costly: only while masked or an open hand (the only kind that swipes) is up near the face
        if soul.masked() or any(all(fingers_out(lm)) and fist_centre(lm)[1] < 0.6 * h for lm in hands):
            face_mesh = face_mesh or mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
            faces = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_face_landmarks
            if faces:
                geometry = face_geometry([(p.x * w, p.y * h) for p in faces[0].landmark])
                xs, ys = geometry[0][:, 0], geometry[0][:, 1]
                face = (ys.min(), ys.max(), xs.min(), xs.max())
        mask = segmenter.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).segmentation_mask
        history.append((frame.copy(), mask))
        for event in soul.update(hands, dt, h, face, body_x(mask)):
            if event in EVENTS:
                play_sound(*EVENTS[event][0])
                if EVENTS[event][1]:
                    shown, shown_until = EVENTS[event][1:], soul.t + 1.2
            if event == "draw":
                particles.burst(soul.hilt, BLUE, count=120, speed=500.0, life=0.6)
            if event == "bankai":
                particles.burst(soul.hilt, RED, count=300, speed=900.0, life=1.2)
            if event == "flash":
                ghosts, flashed = merge_ghosts(list(history)), soul.t
                if battle:
                    battle.sub_until = battle.t + DODGE
            if event == "scatter":
                petals = Petals(soul.hilt, soul.tip())
            if event == "recall":
                particles.burst(soul.hilt, PINK, count=150, speed=400.0, life=0.4)
        if not soul.scattered:
            petals = None
        elif petals is not None:
            petals.update(dt, soul.swarm)
        if battle:
            if battle.update(dt):
                play_sound(200, 80)
            for spot in hunt(battle, soul, dt, petals):
                particles.burst(spot, (240, 240, 240), count=80, speed=450.0, life=0.5)

        if soul.t - flashed < FLASH_TIME:
            draw_afterimages(frame, *ghosts, mask, 1 - (soul.t - flashed) / FLASH_TIME)
        if soul.bankai:
            draw_bankai_aura(frame, mask, soul.t)
        if soul.masked() and geometry is not None:
            age = soul.t - soul.mask_at
            draw_hollow_mask(frame, *geometry, fade=min(1.0, age / 0.3, (MASK_TIME - age) / 1.0))
        for m in battle.monsters if battle else []:
            draw_hollow(frame, m, battle.radius, battle.t)
        if petals is not None:
            draw_petals(frame, petals)
        elif soul.sword_at is not None:
            draw_sword(frame, soul.hilt, soul.tip(), soul.bankai)
        for fired, origin, direction, size, bankai, _ in soul.slashes:
            draw_getsuga(frame, origin, direction, soul.t - fired, size, bankai)
        particles.update(dt)
        particles.draw(frame)

        if shown and soul.t < shown_until:
            banner(frame, *shown)
        state = ("SENBONZAKURA" if soul.scattered else "BANKAI" if soul.bankai else "ZANPAKUTO" if soul.sword_at is not None else "NO SWORD") + \
                (f" + HOLLOW {math.ceil(MASK_TIME - (soul.t - soul.mask_at))}" if soul.masked() else "")
        if battle:
            if battle.t - battle.hurt_time < 0.3:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
            cv2.putText(frame, f"HP {battle.hp}  WAVE {battle.wave}  SCORE {battle.score}", (10, h - 70),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (240, 240, 240), 2)
            if battle.over():
                banner(frame, "THE HOLLOWS WON - R", RED)
        cv2.putText(frame, f"BLEACH  {state}", (10, h - 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 200, 120), 2)
        cv2.putText(frame, "Fist low, pull up: sword | Swing: Getsuga | Grip sword wrist: Bankai, then open hand: petals | "
                    "Open hand down face: mask | Lunge sideways: Flash Step | H: Hollows | M, Q", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (230, 230, 230), 1)
        cv2.imshow("JutsuDevilFruits", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return "quit"
        if key == ord('m'):
            return "menu"
        if key == ord('h'):
            battle = None if battle else Battle(w, h)
        if key == ord('r') and battle and battle.over():
            battle = Battle(w, h)
