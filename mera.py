# mera.py - Mera Mera mode: the flame-flame fruit over the webcam image.
#   Fist: it burns, flames rising off it and following the hand.
#   Punch the fist at the camera (its knuckles grow fast on screen): Hiken, a fire blast rushing at you.
# Only hand landmarks are needed. Mera and burn are pure (no camera) and tested.
import math
import time
import cv2
import numpy as np
from anime_effects import draw_fireball
from effects import ParticleSystem, play_sound
from hand_detector import HandDetector
from one_piece import MAX_DT, gesture

PUNCH_GROWTH = 1.35      # Hiken: knuckle width grows by this factor...
PUNCH_WINDOW = 0.25      # ...within this many seconds (a slow approach doesn't count)
HIKEN_TIME = 1.6         # seconds the fire blast lasts
FLAMES = ((0, 60, 230), (0, 140, 255), (40, 210, 255))   # deep red, orange, yellow (BGR)
FLAME_RISE = 320.0       # px/s: flames go up off the fist, not out


def knuckles(lm):
    """Knuckle width (index to pinky knuckle). Unlike wrist-to-knuckle it doesn't shrink when the
    fist points at the camera, so it grows with a punch."""
    return math.dist(lm[5], lm[17])


def fist_centre(lm):
    return np.mean([lm[k] for k in (0, 5, 9, 13, 17)], axis=0)


class Mera:
    def __init__(self):
        self.t = 0.0
        self.fists = []          # (centre, knuckle width) of each fist this frame
        self.widths = []         # (time, width, centre) of the biggest fist over the last PUNCH_WINDOW s
        self.hiken_at, self.hiken_from, self.hiken_scale = None, None, 1.0

    def update(self, hands, dt):
        """hands: list of 21-landmark pixel lists. Returns True on the frame a Hiken fires."""
        self.t += min(dt, MAX_DT)
        self.fists = [(fist_centre(lm), knuckles(lm)) for lm in hands if gesture(lm) == "fist"]
        if not self.fists:
            return False         # a dropped frame keeps the samples; they age out on their own
        centre, width = max(self.fists, key=lambda f: f[1])
        if self.widths and math.dist(centre, self.widths[-1][2]) > 2 * width:
            self.widths = []     # a different fist: its size says nothing about a punch
        self.widths = [s for s in self.widths if s[0] >= self.t - PUNCH_WINDOW] + [(self.t, width, centre)]
        if self.blasting() or width < PUNCH_GROWTH * min(w for _, w, _ in self.widths):
            return False
        self.hiken_at, self.hiken_from, self.hiken_scale = self.t, centre, min(width / 40, 1.5)
        self.widths = []         # one punch, one Hiken
        return True

    def blasting(self):
        return self.hiken_at is not None and self.t - self.hiken_at < HIKEN_TIME

    def draw_hiken(self, frame):
        if self.blasting():
            draw_fireball(frame, self.hiken_from, self.t - self.hiken_at, HIKEN_TIME, self.hiken_scale, 0.0)


def burn(particles, centre, width, count=8):
    """Flames licking up off a fist: fire-coloured embers kicked upward (the particles add up as light)."""
    for colour in FLAMES:
        particles.burst(centre, colour, count=count, speed=3 * width, life=0.8)
        particles.vel[-count:, 1] -= FLAME_RISE * np.random.uniform(0.5, 1.0, count)


def play(cap):
    """Mera Mera mode loop until Q ("quit") or M ("menu")."""
    detector = HandDetector(max_hands=2)
    mera, particles, last = Mera(), ParticleSystem(), time.monotonic()
    while True:
        success, frame = cap.read()
        if not success:
            print("Camera stopped sending frames.")
            return "quit"
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        _, hands = detector.find_hands(frame, draw=False)
        now = time.monotonic()
        dt, last = now - last, now
        if mera.update([detector.get_landmark_list(lm, frame.shape) for lm in hands], dt):
            play_sound(200, 600)
        for centre, width in mera.fists:
            burn(particles, centre, width)
        particles.update(dt)
        particles.draw(frame)
        mera.draw_hiken(frame)
        if mera.blasting():
            text = "HIKEN!"
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.6, 3)[0]
            cv2.putText(frame, text, ((w - size[0]) // 2, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 90, 255), 3)
        cv2.putText(frame, "MERA MERA", (10, h - 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 140, 255), 2)
        cv2.putText(frame, "Fist: burn | Punch at the camera: Hiken | M, Q", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.imshow("JutsuDevilFruits", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return "quit"
        if key == ord('m'):
            return "menu"
