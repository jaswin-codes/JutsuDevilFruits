# signs.py
# Each sign is a list: [thumb, index, middle, ring, pinky]
# 1 = finger up, 0 = finger down, -1 = ignore (wildcard)

# Thumb is a wildcard everywhere: it's the least reliable finger, and the
# other four fingers already tell every sign apart.
SIGNS = {
    "Tora":  [-1, 1, 1, 0, 0],   # Tiger  - index + middle up
    "Ushi":  [-1, 1, 1, 1, 1],   # Ox     - open hand
    "Ne":    [-1, 1, 0, 0, 0],   # Rat    - index only
    "Inu":   [-1, 0, 0, 0, 0],   # Dog    - fist
    "Tori":  [-1, 1, 0, 0, 1],   # Bird   - index + pinky up
}
'''
Ne → Tora → Inu KATON FIREBALL 🔥
Ushi → Ne → Tora SHADOW CLONE 👥
Tori → Inu → Ushi CHIDORI ⚡
'''

def match_sign(finger_states):
    """
    Compares detected finger states against all defined signs.
    Returns the sign name if matched, or None.
    
    -1 in a sign pattern = wildcard (ignore that finger)
    """
    for sign_name, pattern in SIGNS.items():
        match = True
        for detected, expected in zip(finger_states, pattern):
            if expected == -1:
                continue          # wildcard, skip this finger
            if detected != expected:
                match = False
                break
        if match:
            return sign_name
    return None

def combine_signs(signs):
    """
    Per-hand signs ordered left to right on screen -> one sign name.
    The frame is mirrored, so screen-left is the user's own left hand.
    One hand: its sign as-is. Two hands: "Left+Right", or None if either is unknown.
    """
    if any(sign is None for sign in signs):
        return None
    return "+".join(signs) or None
