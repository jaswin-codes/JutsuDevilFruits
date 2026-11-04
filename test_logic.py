# Webcam-free tests for the recognition pipeline. Run: python -m pytest
import sys
import types

sys.modules.setdefault("mediapipe", types.ModuleType("mediapipe"))  # detector math doesn't need it

from hand_detector import HandDetector
from signs import match_sign, combine_signs
from stabilizer import SignStabilizer
from sequence import SequenceDetector
import combos
from combos import match_combo


def peace_sign_landmarks():
    """Upright hand, wrist at bottom: index + middle extended, ring + pinky curled."""
    lm = [(0, 0)] * 21
    lm[0], lm[3], lm[4], lm[17] = (100, 300), (60, 240), (70, 250), (150, 220)
    for tip, pip, x, extended in [(8, 6, 80, True), (12, 10, 100, True),
                                  (16, 14, 120, False), (20, 18, 140, False)]:
        lm[pip] = (x, 180)
        lm[tip] = (x, 120) if extended else (x, 230)
    return lm


def test_finger_states_survive_rotation():
    lm = peace_sign_landmarks()
    rotated = [(300 - y, x) for x, y in lm]  # 90 degrees
    upside_down = [(-x, -y) for x, y in lm]
    for hand in (lm, rotated, upside_down):
        assert HandDetector.get_finger_states(None, hand)[1:] == [1, 1, 0, 0]


def test_thumb_is_ignored():
    for thumb in (0, 1):
        assert match_sign([thumb, 1, 1, 0, 0]) == "Tora"
        assert match_sign([thumb, 1, 1, 1, 1]) == "Ushi"
        assert match_sign([thumb, 1, 0, 0, 0]) == "Ne"
        assert match_sign([thumb, 0, 0, 0, 0]) == "Inu"
        assert match_sign([thumb, 1, 0, 0, 1]) == "Tori"


def test_full_combo_through_pipeline():
    stab, seq = SignStabilizer(), SequenceDetector()
    t = 0.0
    for sign in ["Ne"] * 10 + [None] * 3 + ["Tora"] * 10 + ["Inu"] * 10:
        t += 1 / 30
        stable = stab.update(sign)
        if stable:
            result = seq.update(stable, now=t)
    assert match_combo(result)["name"].startswith("KATON")


def test_sequence_expires_after_long_gap():
    seq = SequenceDetector(timeout=2.0)
    seq.update("Ne", now=0.0)
    seq.update("Ne", now=5.0)            # holding a sign doesn't expire it
    assert seq.update("Tora", now=6.0) == ["Ne", "Tora"]
    assert seq.update("Inu", now=9.0) == ["Inu"]  # 3s gap: stale signs dropped


def test_combos_of_any_length_match_on_sequence_end(monkeypatch):
    monkeypatch.setitem(combos.COMBOS, ("Ushi", "Inu"), {"name": "TWO", "colour": (0, 0, 0)})
    monkeypatch.setitem(combos.COMBOS, ("Tori", "Ushi", "Inu"), {"name": "THREE", "colour": (0, 0, 0)})
    assert match_combo(["Ne", "Ushi", "Inu"])["name"] == "TWO"
    assert match_combo(["Tori", "Ushi", "Inu"])["name"] == "THREE"  # longest match wins
    assert match_combo(["Inu"]) is None


def test_two_hand_signs_combine_left_to_right():
    assert combine_signs(["Tora"]) == "Tora"          # one hand unchanged
    assert combine_signs(["Ushi", "Tora"]) == "Ushi+Tora"
    assert combine_signs(["Ushi", None]) is None      # either hand unknown -> no sign
    assert combine_signs([]) is None
    assert match_combo(["Ushi+Ushi", "Tora+Tora"])["name"] == "RASENGAN"
