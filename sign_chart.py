# sign_chart.py - pictures of every hand sign: the full chart (menu 0, H mid-game) and the
# small row of target signs in the challenge HUD. Pictures: assets/signs (see SOURCE.md there).
import os
import time

import cv2
import numpy as np

from sign_model import LABELS

SIGNS = [s for s in LABELS if s]   # the 12 zodiac signs + Gassho, in model order
MEANING = {"Ne": "Rat", "Ushi": "Ox", "Tora": "Tiger", "U": "Hare", "Tatsu": "Dragon", "Mi": "Snake",
           "Uma": "Horse", "Hitsuji": "Ram", "Saru": "Monkey", "Tori": "Bird", "Inu": "Dog", "I": "Boar",
           "Gassho": "Prayer"}
SIGN_DIR = os.path.join(os.path.dirname(__file__), "assets", "signs")
_pictures = {}


def picture(sign, width):
    """The sign's photo scaled to width (4:3), cached."""
    if (sign, width) not in _pictures:
        img = cv2.imread(os.path.join(SIGN_DIR, f"{sign}.jpg"))
        _pictures[sign, width] = cv2.resize(img, (width, width * 3 // 4), interpolation=cv2.INTER_AREA)
    return _pictures[sign, width]


def paste(frame, img, x, y):
    """Copies img onto frame with its top-left at (x, y), clipped to the frame."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if x0 < x1 and y0 < y1:
        frame[y0:y1, x0:x1] = img[y0 - y:y1 - y, x0 - x:x1 - x]


def chart_cells(n, w, h, cols=5, top=50, label=32):
    """(x, y, picture width) of n cells in a cols-wide grid under a title, filling a w x h frame."""
    rows = -(-n // cols)
    cell_w = w // cols
    pic_w = min(cell_w - 8, int(((h - top - 10) / rows - label) * 4 / 3))
    pad = (w - cols * cell_w) // 2 + (cell_w - pic_w) // 2
    step_y = pic_w * 3 // 4 + label
    return [(pad + (k % cols) * cell_w, top + (k // cols) * step_y, pic_w) for k in range(n)]


def draw_chart(frame, footer):
    """Every sign with its name, over the dimmed frame (in place)."""
    h, w = frame.shape[:2]
    frame[:] = (frame * 0.25).astype(np.uint8)
    cv2.putText(frame, "HAND SIGNS", (w // 2 - 105, 36), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 165, 255), 2)
    for sign, (x, y, pic_w) in zip(SIGNS, chart_cells(len(SIGNS), w, h - 20)):
        paste(frame, picture(sign, pic_w), x, y)
        label = f"{sign} - {MEANING[sign]}"
        cv2.putText(frame, label, (x, y + pic_w * 3 // 4 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, footer, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)


def draw_sign_strip(frame, signs, x, y, width=64):
    """Small pictures of signs in a row from (x, y), shrunk to fit the frame width (in place)."""
    gap = 6
    width = min(width, (frame.shape[1] - x - 10) // max(len(signs), 1) - gap)
    for k, sign in enumerate(signs):
        px = x + k * (width + gap)
        paste(frame, picture(sign, width), px, y)
        cv2.rectangle(frame, (px, y), (px + width, y + width * 3 // 4), (255, 255, 255), 1)


def show_chart(cap, footer="Any key or M goes back, Q quits", close_keys=None):
    """Shows the chart over the live camera until a key in close_keys (any key if None) or Q.
    Returns the seconds it was open, or "quit" on Q or a dead camera."""
    start = time.monotonic()
    while True:
        success, frame = cap.read()
        if not success:
            print("Camera stopped sending frames.")
            return "quit"
        frame = cv2.flip(frame, 1)
        draw_chart(frame, footer)
        cv2.imshow("Jutsu Code", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return "quit"
        if key != 0xFF and (close_keys is None or key in close_keys):
            return time.monotonic() - start
