import numpy as np
import one_piece
from one_piece import Gomu, Spring, gesture, stretch

W, H = 640, 480

# Synthetic right hand, fingers up, wrist at (300, 400); palm size (wrist to middle knuckle) 60 px
OPEN = [(300, 400),
        (270, 390), (255, 375), (245, 360), (240, 345),     # thumb
        (285, 340), (283, 315), (282, 300), (281, 285),     # index
        (300, 340), (300, 312), (300, 296), (300, 280),     # middle
        (315, 342), (316, 318), (317, 303), (318, 290),     # ring
        (328, 348), (330, 330), (331, 318), (332, 308)]     # pinky


def fist(dx=0):
    lm = list(OPEN)
    for tip in (8, 12, 16, 20):
        lm[tip] = (lm[tip - 3][0], 365)                     # tips curled back below the knuckles
    return [(x + dx, y) for x, y in lm]


def pinch(dx=0, dy=0):
    lm = list(OPEN)
    lm[4] = (281, 290)                                      # thumb tip on the index tip
    return [(x + dx, y + dy) for x, y in lm]


def test_gesture():
    assert gesture(OPEN) is None
    assert gesture(fist()) == "fist"
    assert gesture(pinch()) == "pinch"
    near = list(OPEN)
    near[4] = (281, 285 + 22)          # gap 22 px = 0.37 palm: between the grab and let-go thresholds
    assert gesture(near) is None and gesture(near, pinching=True) == "pinch"


def gradient():
    xs = np.tile(np.arange(W, dtype=np.uint8)[None, :] // 3, (H, 1))
    return np.dstack([xs, xs, xs])


def test_stretch_moves_grab_to_tip_without_folding():
    frame = gradient()
    frame[195:206, 195:206] = (0, 0, 255)       # red spot at the grab point
    before = frame.copy()
    stretch(frame, (200, 200), (400, 200), 20, back=30)
    assert tuple(frame[200, 400]) == (0, 0, 255)            # the spot is now at the tip
    assert (frame[50] == before[50]).all()                  # far from the band: untouched
    assert (frame[200, :160] == before[200, :160]).all()    # behind the stretched strip: untouched
    line = frame[200, 170:395, 2].astype(int)    # stretched strip: grey ramp then the red spot, no fold back
    assert (np.diff(line) >= 0).all()


def test_stretch_zero_length_is_a_no_op():
    frame = gradient()
    before = frame.copy()
    stretch(frame, (200, 200), (200.4, 200), 20)
    stretch(frame, (-500, -500), (-400, -500), 20)          # entirely off screen
    assert (frame == before).all()


def test_spring_overshoots_then_settles():
    s = Spring()
    s.x = np.array([100.0, 0.0])
    xs = []
    for _ in range(60):
        s.step(np.zeros(2), 1 / 30)
        xs.append(s.x[0])
    assert min(xs) < -5                       # wobbles past the rest point: rubber
    assert abs(xs[-1]) < 2


def run(g, hands, seconds):
    out = []
    for _ in range(int(seconds * 30)):
        out = g.update(hands, 1 / 30)
    return out


def test_fist_shoots_out_and_stays_in_frame():
    g = Gomu(W, H)
    (grab, tip, radius, back), = run(g, [fist()], 1.0)
    assert g.pistol()
    assert tip[1] < grab[1] - 2 * 60                        # flew up, the way the hand points
    assert tip[1] - 60 >= -1                                # the fist stays inside the frame
    assert run(g, [OPEN], 1.0) == [] and not g.pistol()    # open hand: snapped back


def test_pinch_pull_goes_further_than_the_fingers_then_snaps_back():
    g = Gomu(W, H)
    g.update([pinch()], 1 / 30)
    out = None
    for k in range(1, 11):                                  # pull 100 px to the right
        out = g.update([pinch(dx=10 * k)], 1 / 30)
    (grab, tip, radius, back), = out
    assert np.allclose(tip - grab, (one_piece.PULL_GAIN * 100, 0))
    assert run(g, [], 1.5) == [] and g.grips == []           # hand gone: snapped back, forgotten


def test_a_dropped_frame_keeps_the_grab():
    g = Gomu(W, H)
    run(g, [fist()], 1.0)
    (_, far, _, _), = g.update([], 1 / 30)                   # tracking lost the hand for two frames
    g.update([], 1 / 30)
    (grab, tip, _, _), = g.update([fist()], 1 / 30)
    assert np.allclose(tip, far, atol=5) and tip[1] < grab[1] - 2 * 60   # still out, no regrow

    g = Gomu(W, H)
    g.update([pinch()], 1 / 30)
    g.update([pinch(dx=40)], 1 / 30)
    src = g.grips[0].src.copy()
    g.update([OPEN], 1 / 30)                                 # one misread frame
    g.update([pinch(dx=50)], 1 / 30)
    assert np.allclose(g.grips[0].src, src)                  # same grab point, pull goes on
