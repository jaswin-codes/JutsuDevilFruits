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


def test_gatling_needs_fast_pumps_not_jitter():
    rng = np.random.default_rng(0)
    g = Gomu(W, H)
    for _ in range(60):                                      # a still fist with tracking jitter
        g.update([fist(dx=int(rng.integers(-3, 4)))], 1 / 30)
    assert not g.gatling()

    swing = [0, 20, 40, 20]                                  # 0.67 palm swings, 15 reversals a second
    for k in range(30):
        out = g.update([fist(dx=swing[k % 4])], 1 / 30)
    assert g.gatling() and len(out) == 1 + one_piece.GATLING_FISTS
    assert len({tuple(np.round(tip)) for _, tip, _, _ in out}) == len(out)   # fanned out, all different
    run(g, [fist(dx=swing[3])], 1.0)                         # stopped pumping
    assert not g.gatling() and g.pistol()


def test_gatling_fists_all_copy_the_fist():
    g = Gomu(W, H)
    swing = [0, 20, 40, 20]
    for k in range(30):
        out = g.update([fist(dx=swing[k % 4])], 1 / 30)
    frame = gradient()
    grab = np.round(out[0][0]).astype(int)
    frame[grab[1] - 5:grab[1] + 6, grab[0] - 5:grab[0] + 6] = (0, 0, 255)   # red spot on the fist
    clean = frame.copy()
    for grab_, tip, radius, back in out:
        stretch(frame, grab_, tip, radius, back, image=clean)
    red = [tip for _, tip, _, _ in out
           if 0 <= tip[0] < W and 0 <= tip[1] < H and tuple(frame[int(tip[1]), int(tip[0])]) == (0, 0, 255)]
    assert len(red) >= 3 and red[-1] is out[-1][1]          # several fists (some overlap), the main one on top;
    #                                                          without image=clean every later fist copies arm: 0


def two():
    lm = list(OPEN)
    for tip in (16, 20):
        lm[tip] = (lm[tip - 3][0], 365)                     # ring and pinky curled: V sign
    return lm


def test_gear_second_sign_speeds_up_the_pistol_then_runs_out():
    assert gesture(two()) == "two"
    g = Gomu(W, H)
    run(g, [two()], 0.3)
    assert g.gear is None                                    # not held long enough yet
    out = run(g, [two()], 0.4)
    assert g.gear == 2 and out == []                         # switched on; a sign draws no stretch
    (grab, fast, _, _), = g.update([fist()], 1 / 30)
    plain = Gomu(W, H)
    (_, slow, _, _), = plain.update([fist()], 1 / 30)
    assert grab[1] - fast[1] > 2 * (grab[1] - slow[1])       # Pistol flies out much faster
    (_, fast, _, _), = run(g, [fist()], 0.3)
    assert fast[1] - 60 >= -1                                # yet the overshoot stays in the frame
    run(g, [OPEN], one_piece.GEAR_TIME)
    assert g.gear is None


def test_gear_second_tints_only_the_person():
    frame = gradient()
    before = frame.copy()
    mask = np.zeros((H, W), np.float32)
    mask[100:300, 200:400] = 1
    one_piece.gear_second(frame, mask, 0.0)
    assert (frame[:100] == before[:100]).all()
    inside = frame[200, 300].astype(int) - before[200, 300]
    assert inside[2] > 30 and inside[2] > inside[1]          # redder (BGR), more red than green


def four():
    lm = list(OPEN)
    lm[4] = (308, 345)                                      # thumb tucked over the palm
    return lm


def test_gear_fourth_sign_ends_gear_second_and_grows_the_fists():
    assert gesture(four()) == "four" and gesture(OPEN) is None
    g = Gomu(W, H)
    run(g, [two()], 0.7)
    run(g, [four()], 0.7)
    assert g.gear == 4                                       # the newer gear replaces the other
    (_, _, radius, _), = run(g, [fist()], 0.5)
    assert np.isclose(radius, 0.9 * 60 * one_piece.GEAR4_FIST)
    assert len(one_piece.puffs([fist()])) == 2               # fist and forearm
    mask = np.zeros((H, W), np.float32)
    mask[150:450, 200:440] = 1
    (centre, r), = one_piece.puffs([], mask)
    assert 200 < centre[0] < 440 and 150 < centre[1] < 450 and r > 50


def test_bulge_magnifies_the_middle_without_seam_or_fold():
    frame = gradient()
    before = frame.copy()
    one_piece.bulge(frame, (150, 200), 80)               # gradient() wraps at x = 256: stay left of it
    assert (frame[:115] == before[:115]).all() and (frame[290:] == before[290:]).all()   # outside: untouched
    assert (frame[200, :65] == before[200, :65]).all()
    line = frame[200, 70:231, 0].astype(int)
    assert (np.diff(line) >= 0).all()                        # still a ramp: no fold
    assert line[90] - line[70] < before[200, 160, 0] - before[200, 140, 0] - 2   # magnified in the middle
