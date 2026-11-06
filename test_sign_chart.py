import os

import numpy as np

from combos import COMBOS
from game import Game
from sign_chart import SIGNS, SIGN_DIR, MEANING, chart_cells, draw_sign_strip, paste


def test_every_sign_the_game_uses_has_a_picture_and_name():
    used = set(SIGNS) | {s for signs in COMBOS for s in signs}
    assert used == set(SIGNS) and len(SIGNS) == 13
    for s in used:
        assert os.path.exists(os.path.join(SIGN_DIR, f"{s}.jpg")) and s in MEANING


def test_chart_cells_fit_the_frame_without_overlap():
    cells = chart_cells(len(SIGNS), 640, 460)
    boxes = [(x, y, x + pw, y + pw * 3 // 4 + 32) for x, y, pw in cells]   # picture + label
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 and x1 <= 640 and 0 <= y0 and y1 <= 460
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def test_paste_clips_at_the_frame_edge():
    frame = np.zeros((10, 10, 3), np.uint8)
    paste(frame, np.full((4, 4, 3), 255, np.uint8), 8, -2)
    assert frame[:2, 8:].all() and frame.sum() == 255 * 3 * 4


def test_sign_strip_shrinks_long_jutsu_to_fit():
    frame = np.zeros((480, 640, 3), np.uint8)
    draw_sign_strip(frame, ("Mi", "Tora", "Saru", "I", "Uma", "Tora"), 20, 300, width=200)
    assert not frame[:, 631:].any()


def test_pause_moves_the_challenge_deadline():
    game = Game(COMBOS, now=0.0)
    game.pause(5.0)
    assert game.time_left(60.0) == 5.0
