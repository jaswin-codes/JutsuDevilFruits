import numpy as np
from battle import Battle, Monster
from bleach import (BANKAI_HOLD, GETSUGA_TIME, Shinigami, crescent, draw_bankai_aura, draw_getsuga, draw_hollow, hunt,
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


def test_the_sword_follows_a_fist_that_jumps_far_in_one_frame():
    soul = drawn()
    events = flat(run(soul, [[moved(fist(), dx=220, dy=-100)]]))
    assert soul.fist[0] > 500 and "sheathe" not in events       # the sword moved with the fist


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


def test_a_swing_through_dropped_frames_at_a_low_frame_rate_still_slashes():
    soul = drawn()
    frames = [[moved(fist(), dy=-100)], [], [moved(fist(), dx=200, dy=-100)]]   # 10 fps, hand lost mid-swing
    assert flat(run(soul, frames, dt=0.1)) == ["getsuga"]


def test_slashes_are_spaced_out_by_the_cooldown():
    soul = drawn()
    events = flat(run(soul, [[moved(fist(), dx=200 * k, dy=-100)] for k in range(1, 3)]))   # one long fast swing
    assert events == ["getsuga"]


def test_a_turned_fist_reading_as_loose_keeps_the_sword():
    soul = drawn()
    loose = moved(fist(), dy=-100)
    loose[8], loose[12] = (loose[8][0], loose[8][1] - 200), (loose[12][0], loose[12][1] - 200)    # two fingers read out
    assert not flat(run(soul, [[loose]] * 40)) and soul.sword_at is not None


def test_the_sword_finds_the_fist_again_far_from_where_tracking_lost_it():
    soul = drawn()
    assert "sheathe" not in flat(run(soul, [[]] * 6 + [[moved(fist(), dx=450, dy=-100)]] * 40))
    assert soul.fist[0] > 500 and soul.sword_at is not None


def test_an_open_hand_lets_go_of_the_sword():
    soul = drawn()
    assert flat(run(soul, [[moved(OPEN, dy=-100)]] * 40)) == ["sheathe"]


def hollow_at(battle, x, y, hp=100):
    battle.monsters.append(Monster(x, y, -1, hp, 0.0))
    return battle.monsters[-1]


def test_a_flying_slash_cuts_each_hollow_once():
    battle, hit = Battle(W, H), set()
    tough = hollow_at(battle, 400, 300, hp=250)
    for _ in range(3):
        assert battle.slash((400, 300), 50, 100, hit) == []
    assert tough.hp == 150


def test_getsuga_flies_into_a_hollow_and_kills_it():
    soul, battle = drawn(), Battle(W, H)
    run(soul, [[moved(fist(), dx=130 * k, dy=-100)] for k in range(1, 3)])
    _, origin, direction, size, _, _ = soul.slashes[0]
    far = (origin + direction * 450).tolist()
    m = hollow_at(battle, *far)
    soul.sword_at = None                                    # only the slash counts here
    assert hunt(battle, soul, 0.0) == [] and m.hp == 100    # the slash has not got there yet
    soul.t = soul.slashes[0][0] + 0.5
    assert hunt(battle, soul, 0.0) == [tuple(far)] and not battle.monsters


def test_the_blade_itself_cuts_a_hollow_it_touches():
    soul, battle = drawn(), Battle(W, H)
    m = hollow_at(battle, *soul.tip().tolist())
    hunt(battle, soul, 0.1)
    assert m.hp < 100


def test_hollow_draws_a_bone_mask_where_it_stands():
    frame = blank()
    draw_hollow(frame, Monster(300, 300, -1, 100, 0.0), 40, 0.0)
    assert frame[280, 300].min() > 200 and (frame[20, 20] == 90).all()
