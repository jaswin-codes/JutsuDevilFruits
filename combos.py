# combos.py
# Each combo: sign sequence -> name shown on screen + effect colour (BGR).
# Combos can be any length. No emoji: OpenCV can't draw them.
COMBOS = {
    ("Ne", "Tora", "Inu"):   {"name": "KATON - FIREBALL JUTSU", "colour": (0, 80, 255)},
    ("Ushi", "Ne", "Tora"):  {"name": "SHADOW CLONE JUTSU",     "colour": (255, 200, 0)},
    ("Tori", "Inu", "Ushi"): {"name": "CHIDORI",                "colour": (0, 255, 255)},
}

MAX_COMBO_LENGTH = max(len(signs) for signs in COMBOS)


def match_combo(sequence):
    """Returns the combo the sequence ends with (longest match wins), or None."""
    for signs in sorted(COMBOS, key=len, reverse=True):
        if tuple(sequence[-len(signs):]) == signs:
            return COMBOS[signs]
    return None
