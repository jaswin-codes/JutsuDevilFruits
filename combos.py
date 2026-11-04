# combos.py
# Each combo: sign sequence -> name shown on screen, effect colour (BGR),
# and sound as a (start Hz, end Hz) sweep.
# Combos can be any length and mix one- and two-hand signs. No emoji: OpenCV can't draw them.
COMBOS = {
    ("Ne", "Tora", "Inu"):   {"name": "KATON - FIREBALL JUTSU", "colour": (0, 80, 255),   "sound": (180, 60)},
    ("Ushi", "Ne", "Tora"):  {"name": "SHADOW CLONE JUTSU",     "colour": (255, 200, 0),  "sound": (300, 600)},
    ("Tori", "Inu", "Ushi"): {"name": "CHIDORI",                "colour": (0, 255, 255),  "sound": (1800, 900)},
    # Two-hand signs are "Left+Right" (your own left and right hand)
    ("Ushi+Ushi", "Tora+Tora"): {"name": "RASENGAN",           "colour": (255, 150, 40), "sound": (400, 1200)},
}

MAX_COMBO_LENGTH = max(len(signs) for signs in COMBOS)


def match_combo(sequence):
    """Returns the combo the sequence ends with (longest match wins), or None."""
    for signs in sorted(COMBOS, key=len, reverse=True):
        if tuple(sequence[-len(signs):]) == signs:
            return COMBOS[signs]
    return None
