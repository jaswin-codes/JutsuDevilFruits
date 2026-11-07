import numpy as np
from bleach import (BANKAI_HOLD, GETSUGA_TIME, Shinigami, crescent, draw_bankai_aura, draw_getsuga,
                    draw_hollow_mask, draw_sword)
from test_one_piece import H, OPEN, W, fist

FACE = (100, 300, 200, 400)          # top, bottom, left, right of a face box


def moved(lm, dx=0, dy=0):
    return [(x + dx, y + dy) for x, y in lm]


def run(soul, frames, dt=0.05, face=None):
    """Events of each frame; face is given on the first frame only (then the hand hides it)."""
    return [soul.update(hands, dt, H, face if k == 0 else None) for k, hands in enumerate(frames)]


def drawn():
    """A Shinigami that has just drawn its sword and held it still past the swing lock."""
    soul = Shinigami()
    run(soul, [[fist()], [moved(fist(), dy=-50)], [moved(fist(), dy=-100)]])
    run(soul, [[moved(fist(), dy=-100)]] * 12)
    return soul


def flat(events):
    return [e for frame in events for e in frame]


def test_pulling_a_low_fist_up_fast_draws_the_sword():
    soul = Shinigami()
    events = run(soul, [[fist()], [moved(fist(), dy=-50)], [moved(fist(), dy=-100)]])
    assert flat(events) == ["draw"] and soul.sword_at is not None
    assert np.allclose(soul.tip(), soul.hilt)    # the blade grows out of the fist
    run(soul, [[moved(fist(), dy=-100)]] * 8)
    direction = soul.tip() - soul.hilt           # along the knuckle row, pinky to index side: left, a bit up
    assert direction[0] < 0 and direction[1] < 0
    assert abs(np.hypot(*(soul.tip() - soul.hilt)) - 4.5 * 60) < 1   # fully grown: 4.5 palm sizes


def test_slow_rise_or_a_high_fist_draws_nothing():
    assert not flat(run(Shinigami(), [[moved(fist(), dy=-10 * k)] for k in range(12)], dt=0.1))
    assert not flat(run(Shinigami(), [[moved(fist(), dy=-200)], [moved(fist(), dy=-300)]]))
    assert not flat(run(Shinigami(), [[OPEN], [moved(OPEN, dy=-100)]]))


def test_swinging_fires_one_getsuga_but_not_right_after_drawing():
    soul = Shinigami()
    run(soul, [[fist()], [moved(fist(), dy=-50)], [moved(fist(), dy=-100)]])
    # the draw goes on fast: inside the swing lock, no slash
    assert not flat(run(soul, [[moved(fist(), dx=130 * k, dy=-100)] for k in range(1, 4)]))
    soul = drawn()
    events = flat(run(soul, [[moved(fist(), dx=130 * k, dy=-100)] for k in range(1, 4)]))
    assert events == ["getsuga"] and len(soul.slashes) == 1
    assert soul.slashes[0][2][0] > 0.99          # it flies the way the blade swung (right)
    run(soul, [[moved(fist(), dx=390, dy=-100)]] * int(GETSUGA_TIME / 0.05 + 1))
    assert not soul.slashes                       # gone once it has flown


def test_small_moves_dont_swing():
    soul = drawn()
    assert not flat(run(soul, [[moved(fist(), dx=20 * k, dy=-100)] for k in range(10)]))


def gripped(steps, drop=()):
    """Frames with the sword fist and the other hand's palm on its wrist (frame k in drop: hands lost)."""
    sword = moved(fist(), dy=-100)
    grip = moved(OPEN, dx=-5, dy=-100 + 46)       # palm centre on the sword wrist
    return [[sword] if k in drop else [sword, grip] for k in range(steps)]


def test_gripping_the_sword_wrist_for_a_second_is_bankai():
    soul = drawn()
    events = flat(run(soul, gripped(int(BANKAI_HOLD / 0.05) + 2, drop={8})))   # one dropped frame is fine
    assert events == ["bankai"] and soul.bankai


def test_short_grips_are_not_bankai():
    soul = drawn()
    for _ in range(3):
        run(soul, gripped(10))                    # 0.5 s grip
        run(soul, [[moved(fist(), dy=-100)]] * 10)
    assert not soul.bankai


def test_the_sword_stays_in_its_hand_when_the_gripping_hand_is_a_fist_too():
    soul = drawn()
    before = soul.fist.copy()
    run(soul, [[moved(fist(), dy=-100 + 46), moved(fist(), dy=-100)]] * 5)
    assert np.allclose(soul.fist, before)


def test_sword_is_sheathed_when_the_fist_is_gone():
    soul = drawn()
    soul.bankai = True
    events = flat(run(soul, [[]] * 40))
    assert events == ["sheathe"] and soul.sword_at is None and not soul.bankai


def test_open_hand_dragged_down_the_face_puts_the_mask_on():
    soul = Shinigami()
    events = run(soul, [[moved(OPEN, dy=-200)], [moved(OPEN, dy=-100)], [OPEN]], face=FACE)
    assert flat(events) == ["mask"] and soul.masked()


def test_upward_slow_or_off_face_swipes_dont_mask():
    up = [[OPEN], [moved(OPEN, dy=-100)], [moved(OPEN, dy=-200)]]
    assert not flat(run(Shinigami(), up, face=FACE))
    slow = [[moved(OPEN, dy=-200 + 20 * k)] for k in range(11)]
    assert not flat(run(Shinigami(), slow, dt=0.1, face=FACE))
    beside = [[moved(OPEN, dx=200, dy=-200)], [moved(OPEN, dx=200)]]
    assert not flat(run(Shinigami(), beside, face=FACE))
    assert not flat(run(Shinigami(), [[moved(OPEN, dy=-200)], [OPEN]]))       # never saw a face


def test_crescent_ends_meet_and_it_is_thickest_in_front():
    shape = crescent((0, 0), (1, 0), 100)
    assert np.hypot(*(shape[0] - shape[-1])) < 2 and np.hypot(*(shape[27] - shape[28])) < 2
    assert abs(shape[:, 0].max() - 100) < 1      # front of the outer arc
    assert shape[:, 0].min() > -30               # the tips curl back only a little


def blank():
    return np.full((H, W, 3), 90, np.uint8)


def test_sword_and_getsuga_draw_light_where_they_are():
    for bankai in (False, True):
        frame = blank()
        draw_sword(frame, (300, 300), (300, 60), bankai)
        assert (frame[200, 300] != 90).any() and (frame[200, 600] == 90).all()
        frame = blank()
        draw_getsuga(frame, (100, 240), (1, 0), 0.2, 80, bankai)
        assert (frame[240, 100 + 180 + 110] != 90).any() and (frame[20, 620] == 90).all()
    frame = blank()
    draw_getsuga(frame, (100, 240), (1, 0), GETSUGA_TIME, 80)
    assert (frame == 90).all()                   # over: nothing drawn


def test_bankai_darkens_the_room_but_not_the_person():
    frame, mask = blank(), np.zeros((H, W), np.float32)
    mask[150:450, 250:390] = 1
    draw_bankai_aura(frame, mask, 0.3)
    assert frame[20, 20].max() < 60 and abs(int(frame[300, 320, 0]) - 90) < 5
    assert frame[148:152, 250:390, 2].max() > 150    # red flames at the edge


def test_hollow_mask_covers_the_face_only():
    frame = blank()
    oval = np.float32([(250, 120), (390, 120), (420, 230), (380, 330), (260, 330), (220, 230)])
    irises = [np.float32((285, 200)), np.float32((355, 200))]
    draw_hollow_mask(frame, oval, irises, np.float32((320, 290)))
    assert frame[160, 320].min() > 150           # bone white on the forehead
    assert frame[200, 285].max() < 20 and frame[200, 289][2] > 200 > frame[200, 289][0]   # black pupil, yellow iris
    assert (frame[20, 20] == 90).all() and (frame[230, 450] == 90).all()          # outside the face untouched
