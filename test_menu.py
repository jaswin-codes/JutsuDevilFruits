# Fake-camera smoke tests: main() with a scripted camera, keyboard, clock and hand detector.
import sys
import time
import cv2
import numpy as np
import pytest
import bleach
import goro
import main
import mera
import one_piece
from test_goro import both
from test_mera import scaled
from test_one_piece import fist


class FakeCap:
    def isOpened(self):
        return True

    def set(self, *_):
        pass

    def read(self):
        return True, np.zeros((480, 640, 3), np.uint8)

    def release(self):
        pass


@pytest.fixture
def run_main(monkeypatch):
    """run_main(argv, keys, hands) -> (sounds, detectors used). keys: one per frame (NUL = no key),
    then Q. hands: module name -> list of per-frame hand lists that module's detector sees."""
    def run(argv, keys, hands=None):
        sounds, used = [], []
        keys, clock = iter(keys), iter(np.arange(0, 1e4, 0.05))

        def detector_for(name):
            frames = iter((hands or {}).get(name, []))

            class FakeDetector:
                def __init__(self, max_hands=1):
                    used.append(name)

                def find_hands(self, frame, draw=True):
                    return frame, next(frames, [])

                def get_landmark_list(self, lm, shape):
                    return lm
            return FakeDetector

        monkeypatch.setattr(sys, "argv", ["main.py"] + argv)
        monkeypatch.setattr(main, "SignModel", lambda: None)
        monkeypatch.setattr(cv2, "VideoCapture", lambda _: FakeCap())
        monkeypatch.setattr(cv2, "imshow", lambda *_: None)
        monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
        monkeypatch.setattr(cv2, "waitKey", lambda _: ord(next(keys, "q")))
        monkeypatch.setattr(time, "monotonic", lambda: next(clock))
        for module in (mera, goro, one_piece, bleach):
            monkeypatch.setattr(module, "HandDetector", detector_for(module.__name__))
            monkeypatch.setattr(module, "play_sound", lambda *a: sounds.append(a))
        main.main()
        return sounds, used
    return run


def frames(n):
    return ["\0"] * n           # frames with no key pressed


def test_one_piece_opens_the_fruit_menu_and_plays_each_new_fruit(run_main):
    keys = (["2", "2"] + frames(6) + ["m"]          # worlds -> fruits -> Mera Mera, then M
            + ["2", "m"]                             # fruits again, M goes back to the worlds
            + ["2", "3"] + frames(12) + ["q"])       # -> Goro Goro, quit
    hands = {"mera": [[fist()], [scaled(fist(), 1.5)]], "goro": [both()] * 12}
    sounds, used = run_main([], keys, hands)
    assert used == ["mera", "goro"]
    assert (200, 600) in sounds and (900, 150) in sounds   # Hiken and El Thor fired


@pytest.mark.parametrize("flag, module", [("--one-piece", "one_piece"), ("--mera", "mera"), ("--goro", "goro"),
                                                ("--bleach", "bleach")])
def test_fruit_flags_skip_the_menus(run_main, flag, module):
    assert run_main([flag], frames(3))[1] == [module]


def test_bleach_is_the_third_world_and_draws_a_sword(run_main):
    rise = [[[(x, y - 50 * k) for x, y in fist()]] for k in range(3)]   # a low fist pulled up fast
    sounds, used = run_main([], ["3"] + frames(5) + ["m", "q"], {"bleach": rise})
    assert used == ["bleach"] and (300, 1400) in sounds
