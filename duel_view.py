# duel_view.py - draws duel.py's HUD onto the webcam frame: centre divider, each player's HP bar
# and cooldowns in their bottom corner (the top holds the sign readouts and jutsu banners),
# hit/block notices and the winner screen.
import math

import cv2

from battle import COOLDOWN, PLAYER_HP
from battle_view import COOLDOWN_LABELS, FONT, _text

BAR_W = 200


def _centred(frame, text, cx, y, scale, colour, max_w=10000):
    """Text centred on cx, shrunk to fit max_w pixels."""
    size = cv2.getTextSize(text, FONT, scale, 2)[0]
    if size[0] > max_w:
        scale *= max_w / size[0]
        size = cv2.getTextSize(text, FONT, scale, 2)[0]
    _text(frame, text, (int(cx - size[0] / 2), y), scale, colour)


def draw_duel(frame, duel, messages=(None, None)):
    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
    for i, f in enumerate(duel.fighters):
        half_x0, cx = i * w // 2, w * (0.25 + 0.5 * i)
        if duel.t - f.hurt_time < 0.3:                          # hit: red edge round your half
            cv2.rectangle(frame, (half_x0 + 6, 6), (half_x0 + w // 2 - 6, h - 6), (0, 0, 255), 12)

        x0 = 20 if i == 0 else w - 20 - BAR_W                   # bottom-left / bottom-right corner
        y0 = h - 40
        frac = f.hp / PLAYER_HP
        cv2.rectangle(frame, (x0, y0), (x0 + BAR_W, y0 + 18), (40, 40, 40), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + int(BAR_W * frac), y0 + 18),
                      (0, int(220 * frac), int(255 * (1 - frac))), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + BAR_W, y0 + 18), (255, 255, 255), 1)
        _text(frame, f"P{i + 1}  HP {math.ceil(f.hp)}", (x0 + 6, y0 + 14), 0.5, (255, 255, 255), 1)

        for j, kind in enumerate(COOLDOWN):                     # ready list above the bar
            left = f.cooldown_left(kind)
            label = f"{COOLDOWN_LABELS[kind]} {'READY' if left == 0 else f'{math.ceil(left)}s'}"
            _text(frame, label, (x0, y0 - 10 - 18 * (len(COOLDOWN) - 1 - j)), 0.42,
                  (100, 255, 100) if left == 0 else (120, 120, 120), 1)

        notice = ("BLOCKED!" if duel.t - f.blocked_time < 0.6 else "SHIELD UP" if f.shielded()
                  else "HIDDEN" if f.hidden() else messages[i])
        if notice and duel.fighting():
            _centred(frame, notice, cx, h - 165, 0.7, (0, 200, 255), w // 2 - 30)

    # Round and score between the two HP bars
    _centred(frame, f"ROUND {duel.round}   {duel.wins[0]} - {duel.wins[1]}", w / 2, h - 26, 0.6, (255, 255, 255))

    champ = duel.match_winner()
    if champ is not None:
        lines = (f"PLAYER {champ + 1} WINS THE MATCH!", f"{max(duel.wins)} - {min(duel.wins)}", "Press R for a rematch")
    elif duel.over():
        lines = (f"PLAYER {duel.winner() + 1} WINS ROUND {duel.round}", "Next round...")
    elif duel.t < duel.fight_at:
        lines = (f"ROUND {duel.round}", str(math.ceil(duel.fight_at - duel.t)))
    elif duel.t - duel.fight_at < 0.8:
        lines = ("FIGHT!",)
    else:
        lines = ()
    for k, line in enumerate(lines):
        _centred(frame, line, w / 2, h // 2 + 50 * k, 1.3 if k == 0 else 1.0,
                 (0, 215, 255) if k == 0 else (255, 255, 255), w - 40)
