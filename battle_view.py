# battle_view.py - draws battle.py's monsters and HUD onto the webcam frame (OpenCV shapes only,
# no full-frame float maths, so it stays cheap next to the sign model).
import math

import cv2
import numpy as np

from battle import COOLDOWN, PLAYER_HP

FONT = cv2.FONT_HERSHEY_SIMPLEX
COOLDOWN_LABELS = {"fireball": "FIREBALL", "chidori": "CHIDORI", "rasengan": "RASENGAN", "clone": "CLONE"}


def _text(frame, text, org, scale, colour, thick=2):
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), thick + 3)   # outline: readable on any background
    cv2.putText(frame, text, org, FONT, scale, colour, thick)


def draw_monster(frame, m, r, t):
    """Original 'shade' monster: horned purple blob with glowing eyes, bobbing as it walks."""
    x = int(m.x)
    y = int(m.y + 0.08 * r * math.sin(t * 9 + m.x * 0.05))   # walking bob
    flash = t - m.hit_time < 0.12
    body, dark = ((255, 255, 255), (200, 200, 200)) if flash else ((110, 30, 90), (60, 10, 50))
    face = -m.side                                          # looks toward the centre
    # Horns, then body, then a darker belly
    for hx in (-0.5, 0.5):
        tip = (x + int((hx + 0.15 * face) * r * 1.4), y - int(1.35 * r))
        base = [(x + int((hx - 0.2) * r), y - int(0.6 * r)), (x + int((hx + 0.2) * r), y - int(0.6 * r))]
        cv2.fillPoly(frame, [np.array([base[0], tip, base[1]])], dark, cv2.LINE_AA)
    cv2.ellipse(frame, (x, y), (int(r), int(r * 0.9)), 0, 0, 360, body, -1, cv2.LINE_AA)
    cv2.ellipse(frame, (x, y + int(0.3 * r)), (int(r * 0.6), int(r * 0.45)), 0, 0, 360, dark, -1, cv2.LINE_AA)
    # Eyes with pupils turned toward the player
    for ex in (-0.35, 0.35):
        c = (x + int((ex + 0.15 * face) * r), y - int(0.2 * r))
        cv2.circle(frame, c, max(int(0.2 * r), 2), (60, 230, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (c[0] + int(0.08 * r * face), c[1]), max(int(0.09 * r), 1), (0, 0, 0), -1, cv2.LINE_AA)
    # Jagged mouth
    mx, my, mw = x + int(0.15 * face * r), y + int(0.25 * r), 0.45 * r
    teeth = [(int(mx - mw + i * mw / 3), my + (int(0.12 * r) if i % 2 else 0)) for i in range(7)]
    cv2.polylines(frame, [np.array(teeth)], False, (230, 230, 230), 2, cv2.LINE_AA)
    # HP bar
    bw = int(1.6 * r)
    top = y - int(1.6 * r)
    cv2.rectangle(frame, (x - bw // 2, top), (x + bw // 2, top + 6), (40, 40, 40), -1)
    cv2.rectangle(frame, (x - bw // 2, top), (x - bw // 2 + int(bw * max(m.hp, 0) / m.max_hp), top + 6),
                  (60, 60, 230), -1)


def draw_battle(frame, battle, message=None):
    """Monsters, HP/wave/score (bottom-left: the top holds the sign readout and jutsu banner),
    cooldowns (bottom-right) and the game-over screen."""
    h, w = frame.shape[:2]
    for m in battle.monsters:
        draw_monster(frame, m, battle.radius, battle.t)

    if battle.t - battle.hurt_time < 0.3:                  # hit: red frame edge
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)

    # Player HP bar
    x0, x1, y0 = 20, 220, h - 66
    frac = battle.hp / PLAYER_HP
    cv2.rectangle(frame, (x0, y0), (x1, y0 + 18), (40, 40, 40), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + int((x1 - x0) * frac), y0 + 18),
                  (0, int(220 * frac), int(255 * (1 - frac))), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y0 + 18), (255, 255, 255), 1)
    _text(frame, f"HP {battle.hp}", (x0 + 6, y0 + 14), 0.5, (255, 255, 255), 1)
    _text(frame, f"WAVE {battle.wave}   SCORE {battle.score}", (x0, y0 + 42), 0.55, (255, 255, 255))

    # Cooldowns
    for i, kind in enumerate(COOLDOWN):
        left = battle.cooldown_left(kind)
        label = f"{COOLDOWN_LABELS[kind]} {'READY' if left == 0 else f'{left:.0f}s'}"
        _text(frame, label, (w - 190, h - 20 - 24 * (len(COOLDOWN) - 1 - i)), 0.5,
              (100, 255, 100) if left == 0 else (120, 120, 120), 1)

    if message:
        size = cv2.getTextSize(message, FONT, 0.9, 2)[0]
        _text(frame, message, ((w - size[0]) // 2, h - 130), 0.9, (0, 200, 255))   # under the face

    if battle.over():
        for i, line in enumerate((f"DEFEATED ON WAVE {battle.wave}", f"SCORE {battle.score}",
                                  "Press R to fight again")):
            scale = 1.2 if i == 0 else 0.8
            size = cv2.getTextSize(line, FONT, scale, 2)[0]
            _text(frame, line, ((w - size[0]) // 2, h // 2 + 45 * i), scale, (0, 0, 255) if i == 0 else (255, 255, 255))
