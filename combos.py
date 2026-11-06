# combos.py
# Each combo: sign sequence -> name shown on screen, effect colour (BGR),
# sound as a (start Hz, end Hz) sweep, and which anime effect to draw (anime_effects.py).
# Combos can be any length. No emoji: OpenCV can't draw them.
COMBOS = {
    # Canon sign orders from the anime, signs named as sign_model.LABELS
    ("Mi", "Tora", "Saru", "I", "Uma", "Tora"): {"name": "KATON - FIREBALL JUTSU", "colour": (0, 80, 255),   "sound": (180, 60),   "effect": "fireball"},
    ("Hitsuji", "Mi", "Tora"):                   {"name": "SHADOW CLONE JUTSU",     "colour": (255, 200, 0),  "sound": (300, 600),  "effect": "clone"},
    ("Ushi", "U", "Saru"):                       {"name": "CHIDORI",                "colour": (0, 255, 255),  "sound": (1800, 900), "effect": "chidori"},
    # Hiding with Camouflage: our own pick from the signs no other jutsu uses
    ("Ne", "Inu", "Tori"):                       {"name": "HIDING WITH CAMOUFLAGE",  "colour": (200, 200, 200), "sound": (900, 200),  "effect": "invisible"},
    # Substitution: Tatsu is the one sign no other jutsu uses, so this can't fire halfway through another
    ("Tatsu", "Ne"):                             {"name": "SUBSTITUTION JUTSU",     "colour": (40, 90, 140),  "sound": (500, 250),  "effect": "substitution"},
    # Summoning: canon order, then slap a palm down (main.py waits for the slap before it fires)
    ("I", "Inu", "Tori", "Saru", "Hitsuji"):     {"name": "SUMMONING JUTSU",        "colour": (90, 200, 230), "sound": (120, 40),   "effect": "summon"},
    # Rasengan has no signs in canon: prayer hands (Gassho) start it
    ("Gassho",):                                 {"name": "RASENGAN",               "colour": (255, 150, 40), "sound": (400, 1200), "effect": "rasengan"},
}

MAX_COMBO_LENGTH = max(len(signs) for signs in COMBOS)


def match_combo(sequence):
    """Returns the combo the sequence ends with (longest match wins), or None."""
    for signs in sorted(COMBOS, key=len, reverse=True):
        if tuple(sequence[-len(signs):]) == signs:
            return COMBOS[signs]
    return None
