import numpy as np
from goro import BOLT_TIME, GRACE, Goro, open_up, pointing
from test_one_piece import OPEN, fist


def point(dx=0):
    lm = fist(dx)
    lm[8] = (OPEN[8][0] + dx, OPEN[8][1])                  # index straight up, the rest curled
    return lm


def upside_down(lm):
    return [(x, 800 - y) for x, y in lm]


def run(goro, frames, dt=0.05):
    return [goro.update(hands, dt) for hands in frames]


def both(dx=200):
    return [OPEN, [(x - dx, y) for x, y in OPEN]]


def test_pointing_and_open_hands():
    assert pointing(point()) and not pointing(OPEN) and not pointing(fist())
    assert not pointing(upside_down(point()))
    assert open_up(OPEN) and not open_up(fist()) and not open_up(upside_down(OPEN))


def test_pointing_crackles_at_the_finger():
    goro = Goro()
    goro.update([point(), OPEN], 0.05)
    (centre, radius), = goro.sparks
    assert centre[1] < OPEN[5][1] and radius > 20
    frame = np.zeros((480, 640, 3), np.uint8)
    goro.draw(frame)
    assert frame[int(centre[1]), int(centre[0])].sum() > 500


def test_raising_both_hands_calls_one_bolt_per_raise():
    goro = Goro()
    assert not any(run(goro, [[OPEN]] * 20))                     # one hand is not enough
    fired = run(goro, [both()] * 20)
    assert fired.count(True) == 1 and fired.index(True) >= 5      # held RAISE_HOLD first
    run(goro, [[]] * (GRACE + 1))
    assert any(run(goro, [both()] * 8))


def test_a_misread_frame_keeps_the_hold():
    goro = Goro()
    assert any(run(goro, [both()] * 3 + [[OPEN]] + [both()] * 3))


def test_bolt_comes_from_the_top_between_the_hands_then_ends():
    goro = Goro()
    run(goro, [both()] * 8)
    x = int(goro.strike[0])
    assert 100 < x < 300
    frame = np.zeros((480, 640, 3), np.uint8)
    goro.draw(frame)
    assert frame[0:20, x - 60:x + 60].max() > 200 and frame[0:20, 600:].max() == 0
    run(goro, [[]] * int(BOLT_TIME / 0.05 + 1))
    assert not goro.striking()
