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
    # Blur at 1/4 size and scale back: same soft bloom for a fraction of the cost
    small = cv2.resize(layer, (max(layer.shape[1] // 4, 1), max(layer.shape[0] // 4, 1)),
                       interpolation=cv2.INTER_AREA)
    glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), max(0.5, half / 48)),
                      (layer.shape[1], layer.shape[0]))
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
    w = frame.shape[1]
    mask = np.clip((mask.astype(np.float32) - 0.3) / 0.4, 0, 1)   # crisp edge
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return
    # Only the person's bounding box (and its shifted copies) is touched: keeps FPS up
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = mask[y0:y1, :, None]
    src = person[y0:y1].astype(np.float32)
    out = frame[y0:y1].astype(np.float32)
    fade = float(np.clip((duration - t) / 0.4, 0, 1))    # all clones vanish in the last 0.4s
    for i, offset in enumerate(CLONE_OFFSETS):
        alpha = float(np.clip((t - i * 0.15) / 0.2, 0, 1)) * fade   # pop in one after another
        dx = int(offset * w)
        tx0, tx1 = max(x0 + dx, 0), min(x1 + dx, w)    # clipped at the frame edge, no wrap
        if alpha <= 0 or tx1 <= tx0:
            continue
        a = m[:, tx0 - dx:tx1 - dx] * alpha
        out[:, tx0:tx1] += (src[:, tx0 - dx:tx1 - dx] - out[:, tx0:tx1]) * a
    out[:, x0:x1] += (src[:, x0:x1] - out[:, x0:x1]) * m[:, x0:x1]   # real user in front
    frame[y0:y1] = out.astype(np.uint8)


def face_aim(points):
    """points: face-mesh landmarks as an (N, 2) array of normalised (x, y).
    Returns (mouth centre, fireball scale 0.7-1.5 from how open the mouth is,
    direction -1 = facing screen-left .. 1 = screen-right, 0 = at the camera)."""
    p = np.asarray(points)
    mouth = (p[13] + p[14]) / 2                                   # inner upper/lower lip
    opening = abs(p[14, 1] - p[13, 1]) / max(abs(p[152, 1] - p[10, 1]), 1e-6)   # / face height
    scale = 0.7 + 0.8 * np.clip(opening / 0.08, 0, 1)
    left, right = sorted((p[234, 0], p[454, 0]))                  # cheek edges on screen
    yaw = (p[1, 0] - left) / max(right - left, 1e-6) * 2 - 1      # nose tip across the face
    direction = 0.0 if abs(yaw) < 0.1 else float(np.clip(yaw * 2.5, -1, 1))   # dead zone
    return mouth, scale, direction


def draw_fireball(frame, mouth, t, duration=2.5, scale=1.0, direction=0.0):
    """Fire Style: a fireball swells at the mouth, then flies where you face and fades.
    direction 0 rushes at the camera (grows); -1/1 flies off to the screen's left/right."""
    h, w = frame.shape[:2]
    if t < 0.5:
        radius = (8 + 32 * t / 0.5) * scale                      # gathering at the lips
    else:
        progress = (t - 0.5) / (duration - 0.5)
        # Toward the camera it mostly grows; sideways it mostly travels
        radius = (40 + (0.75 * h - 40) * progress ** 0.7 * (1 - 0.6 * abs(direction))) * scale
        mouth = (mouth[0] + direction * progress * w * 0.7, mouth[1])
    fade = np.clip((duration - t) / 0.6, 0, 1)
    box = _roi(frame, mouth, int(radius * 1.3))
    if box is None or fade <= 0:
        return
    x0, y0, x1, y1 = box
    # Shade at 1/4 resolution then upscale: the fireball is soft, and full-res costs ~150 ms
    sw, sh = max((x1 - x0) // 4, 1), max((y1 - y0) // 4, 1)
    rng = np.random.default_rng(int(t * 15))                   # flames re-shape 15x a second
    noise = cv2.resize(rng.random((6, 6), np.float32), (sw, sh), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:sh, 0:sw] * 4.0
    d = np.hypot(xx + x0 - mouth[0], yy + y0 - mouth[1]) / radius * (1 + 0.45 * (noise - 0.5))
    # Colour by distance from the centre: white-hot core -> yellow -> orange -> deep red (BGR)
    stops = [0.0, 0.35, 0.65, 1.0]
    colour = np.stack([np.interp(d, stops, [200, 0, 0, 0]),
                       np.interp(d, stops, [255, 220, 110, 20]),
                       np.interp(d, stops, [255, 255, 255, 190])], -1)
    alpha = np.clip((1.05 - d) / 0.3, 0, 1) * fade
    size = (x1 - x0, y1 - y0)
    colour = cv2.resize(colour.astype(np.float32), size)
    alpha = cv2.resize(alpha.astype(np.float32), size)[..., None]
    region = frame[y0:y1, x0:x1]
    region[:] = (region + (colour - region) * alpha).astype(np.uint8)
