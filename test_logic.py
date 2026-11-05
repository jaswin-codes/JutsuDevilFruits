# Webcam-free tests for the recognition pipeline. Run: python -m pytest
import numpy as np

import combos
from combos import match_combo
from sequence import SequenceDetector
from sign_model import LABELS, SignModel, best_sign
from stabilizer import SignStabilizer


def test_full_combo_through_pipeline():
    stab, seq = SignStabilizer(), SequenceDetector()
    t = 0.0
    for sign in ["Ushi"] * 10 + [None] * 3 + ["U"] * 10 + ["Saru"] * 10:
        t += 1 / 30
        stable = stab.update(sign)
        if stable:
            result = seq.update(stable, now=t)
    assert match_combo(result)["name"] == "CHIDORI"


def test_sequence_expires_after_long_gap():
    seq = SequenceDetector(timeout=2.0)
    seq.update("Ne", now=0.0)
    seq.update("Ne", now=5.0)            # holding a sign doesn't expire it
    assert seq.update("Tora", now=6.0) == ["Ne", "Tora"]
    assert seq.update("Inu", now=9.0) == ["Inu"]  # 3s gap: stale signs dropped


def test_combos_of_any_length_match_on_sequence_end(monkeypatch):
    monkeypatch.setitem(combos.COMBOS, ("Ushi", "Inu"), {"name": "TWO", "colour": (0, 0, 0), "sound": (1, 1)})
    monkeypatch.setitem(combos.COMBOS, ("Tori", "Ushi", "Inu"), {"name": "THREE", "colour": (0, 0, 0), "sound": (1, 1)})
    assert match_combo(["Ne", "Ushi", "Inu"])["name"] == "TWO"
    assert match_combo(["Tori", "Ushi", "Inu"])["name"] == "THREE"  # longest match wins
    assert match_combo(["Inu"]) is None


def test_combos_use_only_signs_the_model_knows():
    for signs in combos.COMBOS:
        assert set(signs) <= set(LABELS) - {None}


def test_best_sign_picks_top_confident_detection():
    dets = np.array([[0, 2, 0.70, 10, 20, 110, 220],    # Tora
                     [0, 12, 0.95, 50, 60, 150, 260],   # Gassho, highest score
                     [0, 13, 0.40, 0, 0, 5, 5]], np.float32)
    assert best_sign(dets, 0.5, 0.6) == ("Gassho", (100, 120, 300, 520))   # boxes scaled back
    assert best_sign(dets, 0.5, 0.99) == (None, None)       # below threshold
    assert best_sign(dets[2:], 1.0, 0.1) == (None, None)    # "Unknown" class is no sign
    assert best_sign(np.empty((0, 7)), 1.0, 0.1) == (None, None)


def test_sign_model_runs_on_a_frame():
    assert SignModel().detect(np.zeros((480, 640, 3), np.uint8)) == (None, None)


def test_every_combo_has_effect_fields():
    for combo in combos.COMBOS.values():
        assert set(combo) == {"name", "colour", "sound", "effect"}
        assert combo["effect"] in {"chidori", "rasengan", "clone", "fireball", "invisible"}


def test_particles_burst_move_and_expire():
    import numpy as np
    from effects import ParticleSystem
    p = ParticleSystem()
    p.burst((320, 240), (0, 80, 255), count=50)
    p.update(0.1)
    assert len(p.life) == 50 and not np.allclose(p.pos, (320, 240))
    frame = np.zeros((480, 640, 3), np.uint8)
    p.draw(frame)
    assert frame.any()
    p.update(5.0)
    assert len(p.life) == 0


def test_game_scores_only_target_and_ends_on_time():
    from game import Game
    game = Game(combos.COMBOS, now=0.0, duration=60.0)
    target = game.target
    wrong = next(c for signs, c in combos.COMBOS.items() if signs != target)
    assert not game.on_combo(wrong, now=1.0) and game.score == 0
    assert game.on_combo(combos.COMBOS[target], now=2.0) and game.score == 1
    assert game.target != target                       # new target each time
    assert game.time_left(30.0) == 30.0
    assert not game.on_combo(combos.COMBOS[game.target], now=61.0)  # no points after time up
    assert game.over(61.0) and game.score == 1


def test_anime_effects_draw_safely_at_frame_edges():
    import numpy as np
    import anime_effects as fx
    for centre in [(0, 0), (639, 479), (320, 240), (-50, 900)]:
        frame = np.full((480, 640, 3), 40, np.uint8)
        fx.draw_chidori(frame, centre, 40, 0.3)
        fx.draw_rasengan(frame, centre, 40, 0.3)
        for t in (0.2, 1.5, 2.5):
            fx.draw_fireball(frame, centre, t)
        if centre == (320, 240):
            assert frame.max() == 255   # something bright was drawn


def test_clones_copy_person_beside_them():
    import numpy as np
    import anime_effects as fx
    person = np.zeros((480, 640, 3), np.uint8)
    person[200:300, 300:340] = 200                  # a "person" in the middle
    mask = (person[..., 0] > 0).astype(np.float32)
    frame = person.copy()
    fx.draw_clones(frame, person, mask, t=1.0)
    clone_x = 320 + int(fx.CLONE_OFFSETS[3] * 640)
    assert frame[250, clone_x] .min() > 150         # a clone appeared to the right
    assert (frame[250, 320] == 200).all()           # real person untouched


def face_points(nose_x=0.5, lip_gap=0.0):
    """Face-mesh points with only the landmarks face_aim reads filled in."""
    p = np.zeros((468, 2))
    p[10], p[152] = (0.5, 0.2), (0.5, 0.7)            # forehead, chin: face height 0.5
    p[234], p[454] = (0.4, 0.45), (0.6, 0.45)         # cheek edges
    p[1] = (nose_x, 0.45)                             # nose tip
    p[13], p[14] = (0.5, 0.6), (0.5, 0.6 + lip_gap)   # inner lips
    return p


def test_face_aim_reads_head_turn_and_mouth_opening():
    from anime_effects import face_aim
    mouth, scale, direction = face_aim(face_points())
    assert np.allclose(mouth, (0.5, 0.6)) and scale == 0.7 and direction == 0.0
    assert face_aim(face_points(nose_x=0.58))[2] == 1.0     # nose near the right cheek
    assert face_aim(face_points(nose_x=0.45))[2] < 0        # turned left
    assert face_aim(face_points(lip_gap=0.05))[1] == 1.5    # mouth wide open: biggest


def test_fireball_flies_toward_the_side_you_face():
    import anime_effects as fx
    for direction, side in [(1.0, slice(400, 640)), (-1.0, slice(0, 240))]:
        frame = np.zeros((480, 640, 3), np.uint8)
        fx.draw_fireball(frame, (320, 240), t=1.5, direction=direction)
        assert frame[:, side].sum() > 3 * frame[:, 640 - side.stop:640 - side.start].sum()


def test_invisibility_shows_learned_background_where_the_person_stands():
    from anime_effects import BackgroundPlate, draw_invisible
    scene = np.zeros((60, 80, 3), np.uint8)
    scene[:, ::4] = 200                                 # textured wall
    def person_at(x0):
        f, m = scene.copy(), np.zeros((60, 80), np.float32)
        m[10:, x0:x0 + 20] = 1
        f[m > 0] = (0, 0, 255)
        return f, m
    plate = BackgroundPlate()
    plate.update(*person_at(50))                        # first frame includes the person...
    plate.update(*person_at(5))                         # ...whose old spot is now taken whole
    frame, mask = person_at(30)
    assert plate.seen[10:, 30:50].all()
    draw_invisible(frame, mask, plate, t=2.0)
    # wall shows through (compare averages: the ripple shifts the stripes a few pixels)
    assert np.abs(frame[15:, 32:48].mean((0, 1)) - scene[15:, 32:48].mean((0, 1))).max() < 15
    assert not (frame[30, 32:48] == (0, 0, 255)).all(-1).any()
