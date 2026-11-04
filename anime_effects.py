# anime_effects.py - procedural anime-style jutsu effects (numpy + OpenCV only).
# Every function draws into `frame` in place from plain inputs (points, masks, time `t`
# in seconds since the jutsu fired), so they can be rendered and tested without a webcam.
import cv2
import numpy as np


def _roi(frame, centre, half):
    """Frame box of +-half around centre, clamped to the frame: (x0, y0, x1, y1) or None."""
    h, w = frame.shape[:2]
    cx, cy = int(centre[0]), int(centre[1])
    x0, y0, x1, y1 = max(cx - half, 0), max(cy - half, 0), min(cx + half, w), min(cy + half, h)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _add_light(frame, centre, half, draw, bloom=2):
    """Calls draw(layer, (cx, cy)) on a black layer around centre, blooms it, adds it as light."""
    box = _roi(frame, centre, half)
    if box is None:
        return
    x0, y0, x1, y1 = box
    layer = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    draw(layer, (int(centre[0]) - x0, int(centre[1]) - y0))
    glow = cv2.GaussianBlur(layer, (0, 0), max(2, half / 12))
    region = frame[y0:y1, x0:x1]
    cv2.add(region, layer, dst=region)
    for _ in range(bloom):    # a strong bloom sells "light"
        cv2.add(region, glow, dst=region)


def _bolt(rng, start, end, depth=4, spread=0.35):
    """Jagged lightning polyline from start to end by random midpoint displacement."""
    pts = [np.array(start, float), np.array(end, float)]
    for _ in range(depth):
        out = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            d = b - a
            normal = np.array([-d[1], d[0]])
            out += [(a + b) / 2 + normal * rng.uniform(-spread, spread), b]
        pts = out
    return np.int32(pts)


def draw_chidori(frame, centre, radius, t):
    """Crackling silver/white-blue lightning ball held in the palm."""
    rng = np.random.default_rng(int(t * 25))    # new bolts 25x a second = flicker
    radius = max(8, int(radius))

    def draw(layer, c):
        core = int(radius * (0.45 + 0.08 * np.sin(t * 40)))
        cv2.circle(layer, c, int(core * 1.4), (255, 110, 20), -1)    # blue halo (BGR)
        cv2.circle(layer, c, core, (255, 230, 200), -1)
        cv2.circle(layer, c, int(core * 0.6), (255, 255, 255), -1)
        for _ in range(12):   # bolts lashing out of the ball
            angle = rng.uniform(0, 2 * np.pi)
            length = radius * rng.uniform(1.2, 3.0)
            end = (c[0] + np.cos(angle) * length, c[1] + np.sin(angle) * length)
            pts = _bolt(rng, c, end)
            cv2.polylines(layer, [pts], False, (255, 120, 30), 5, cv2.LINE_AA)
            cv2.polylines(layer, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
        for _ in range(14):   # small sparks around the ball
            angle, dist = rng.uniform(0, 2 * np.pi), radius * rng.uniform(0.6, 1.6)
            p = (int(c[0] + np.cos(angle) * dist), int(c[1] + np.sin(angle) * dist))
            cv2.circle(layer, p, 2, (255, 255, 255), -1)

    _add_light(frame, centre, int(radius * 3.2), draw)


def draw_rasengan(frame, centre, radius, t):
    """Spinning blue/white chakra sphere: bright core, swirling arms, a ring of wind."""
    radius = max(8, int(radius))
    spin = t * 14    # radians; fast enough to read as a blur of rotation

    def draw(layer, c):
        for r, colour in [(1.0, (150, 50, 0)), (0.8, (200, 90, 10)),
                          (0.5, (230, 160, 80)), (0.25, (255, 240, 220))]:
            cv2.circle(layer, c, int(radius * r), colour, -1, cv2.LINE_AA)
        s = np.linspace(0, 1, 30)
        for arm in range(6):    # spiral arms from core to rim, rotating
            theta = spin + arm * np.pi / 3 + s * 2.5 * np.pi
            pts = np.int32(np.stack([c[0] + np.cos(theta) * s * radius,
                                     c[1] + np.sin(theta) * s * radius], 1))
            cv2.polylines(layer, [pts], False, (255, 190, 110), 2, cv2.LINE_AA)
        for k in range(4):      # wind streaks orbiting just outside the ball
            start = np.degrees(-spin * 1.3) + k * 90
            cv2.ellipse(layer, c, (int(radius * 1.35), int(radius * 1.15)), 20, start, start + 50,
                        (255, 170, 80), 2, cv2.LINE_AA)

    _add_light(frame, centre, int(radius * 2), draw, bloom=1)


CLONE_OFFSETS = (-0.52, 0.52, -0.28, 0.28)   # fraction of frame width; outer first = behind


def draw_clones(frame, person, mask, t, duration=3.0):
    """Shadow clones: copies of the segmented person beside them, real person kept in front.

    person: clean camera frame (no HUD/landmarks); mask: float HxW, 1 = person.
    """
    h, w = frame.shape[:2]
    mask = np.clip((mask.astype(np.float32) - 0.3) / 0.4, 0, 1)[..., None]   # crisp edge
    src = person.astype(np.float32)
    out = frame.astype(np.float32)
    fade = np.clip((duration - t) / 0.4, 0, 1)    # all clones vanish in the last 0.4s
    for i, offset in enumerate(CLONE_OFFSETS):
        alpha = np.clip((t - i * 0.15) / 0.2, 0, 1) * fade   # pop in one after another
        if alpha <= 0:
            continue
        shift = np.float32([[1, 0, offset * w], [0, 1, 0]])
        clone = cv2.warpAffine(src, shift, (w, h))           # border 0: no wrap-around
        a = cv2.warpAffine(mask, shift, (w, h))[..., None] * alpha
        out = out * (1 - a) + clone * a
    out = out * (1 - mask) + src * mask    # real user stays in front of the clones
    frame[:] = out.astype(np.uint8)
