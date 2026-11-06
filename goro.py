# goro.py - Goro Goro mode: the rumble-rumble (lightning) fruit over the webcam image.
#   Point one finger up: lightning crackles around the hand.
#   Raise both open hands and hold them up: El Thor, a bolt from the top of the frame down between them.
# Only hand landmarks are needed. pointing/open_up/Goro are pure (no camera) and tested.
import math
import time
import cv2
import numpy as np
from anime_effects import draw_bolt, draw_chidori
from effects import play_sound
from hand_detector import HandDetector
from one_piece import GRACE, MAX_DT, fingers_out

POINT_UP = 0.5           # pointing: index tip at least this many palm sizes above its knuckle
RAISE_HOLD = 0.3         # seconds both open hands must stay up to call the bolt
BOLT_TIME = 1.0          # seconds the bolt lasts


def _palm(lm):
    return max(math.dist(lm[0], lm[9]), 1.0)


def pointing(lm):
    """Index finger alone, pointing up."""
    return fingers_out(lm) == [True, False, False, False] and lm[5][1] - lm[8][1] > POINT_UP * _palm(lm)


def open_up(lm):
    """Open hand, fingers up above the wrist."""
    return all(fingers_out(lm)) and lm[12][1] < lm[0][1]


class Goro:
    def __init__(self):
        self.t = 0.0
        self.sparks = []         # (centre, radius) of each pointing hand this frame
        self.held, self.missed, self.armed = 0.0, 0, True   # both hands up: for how long; misread frames
        self.bolt_at, self.strike = None, None

    def update(self, hands, dt):
        """hands: list of 21-landmark pixel lists. Returns True on the frame the bolt fires."""
        dt = min(dt, MAX_DT)
        self.t += dt
        self.sparks = [(np.mean([lm[5], lm[8]], axis=0), 0.6 * _palm(lm)) for lm in hands if pointing(lm)]
        up = [lm for lm in hands if open_up(lm)]
        if len(up) < 2:
            self.missed += 1
            if self.missed > GRACE:          # hands really down: the next raise may call a bolt again
                self.held, self.armed = 0.0, True
            return False
        self.missed = 0
        self.held += dt
        if not self.armed or self.held < RAISE_HOLD:
            return False
        self.armed, self.bolt_at = False, self.t   # one bolt per raise
        self.strike = np.mean([np.mean([lm[0], lm[9]], axis=0) for lm in up[:2]], axis=0)
        return True

    def striking(self):
        return self.bolt_at is not None and self.t - self.bolt_at < BOLT_TIME

    def draw(self, frame):
        for centre, radius in self.sparks:
            draw_chidori(frame, centre, radius, self.t)
        if self.striking():
            draw_bolt(frame, (self.strike[0], 0), self.strike, self.t)


def play(cap):
    """Goro Goro mode loop until Q ("quit") or M ("menu")."""
    detector = HandDetector(max_hands=2)
    goro, last, was_sparking = Goro(), time.monotonic(), False
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
        if goro.update([detector.get_landmark_list(lm, frame.shape) for lm in hands], dt):
            play_sound(900, 150)
        if goro.sparks and not was_sparking:
            play_sound(600, 1200)
        was_sparking = bool(goro.sparks)
        goro.draw(frame)
        text = "EL THOR!" if goro.striking() else "VARI!" if goro.sparks else None
        if text:
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.6, 3)[0]
            cv2.putText(frame, text, ((w - size[0]) // 2, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 220, 120), 3)
        cv2.putText(frame, "GORO GORO", (10, h - 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 200, 80), 2)
        cv2.putText(frame, "Point a finger up: Vari | Raise both open hands: El Thor | M, Q", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.imshow("JutsuDevilFruits", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return "quit"
        if key == ord('m'):
            return "menu"
