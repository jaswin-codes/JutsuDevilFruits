# effects.py - particle bursts and jutsu sounds (numpy + standard library only)
import os
import tempfile
import wave

import cv2
import numpy as np

try:
    import winsound   # Windows only; elsewhere sounds are skipped
except ImportError:
    winsound = None


class ParticleSystem:
    def __init__(self):
        self.pos = np.empty((0, 2))
        self.vel = np.empty((0, 2))
        self.life = np.empty(0)        # seconds left
        self.max_life = np.empty(0)
        self.colour = np.empty((0, 3))

    def burst(self, centre, colour, count=250, speed=700.0, life=1.4):
        angles = np.random.uniform(0, 2 * np.pi, count)
        speeds = np.random.uniform(0.05, 1.0, count) * speed
        vel = np.stack([np.cos(angles), np.sin(angles)], axis=1) * speeds[:, None]
        lives = np.random.uniform(0.5, 1.0, count) * life
        self.pos = np.vstack([self.pos, np.tile(centre, (count, 1)).astype(float)])
        self.vel = np.vstack([self.vel, vel])
        self.life = np.concatenate([self.life, lives])
        self.max_life = np.concatenate([self.max_life, lives])
        self.colour = np.vstack([self.colour, np.tile(colour, (count, 1))])

    def update(self, dt):
        self.pos += self.vel * dt
        self.vel *= 0.92 ** (dt * 30)    # drag, frame-rate independent
        self.vel[:, 1] -= 120 * dt       # drift upward like sparks/flame
        self.life -= dt
        alive = self.life > 0
        for name in ("pos", "vel", "life", "max_life", "colour"):
            setattr(self, name, getattr(self, name)[alive])

    def draw(self, frame):
        if not len(self.life):
            return
        glow = np.zeros_like(frame)
        for (x, y), life, max_life, colour in zip(self.pos, self.life, self.max_life, self.colour):
            radius = max(1, int(9 * life / max_life))
            cv2.circle(glow, (int(x), int(y)), radius, tuple(int(c) for c in colour), -1)
        cv2.add(frame, glow, dst=frame)    # additive blend reads as light


def _sweep_wav(start_hz, end_hz, seconds=0.6, rate=22050):
    """Writes (once) a fading frequency sweep with a little noise, returns its path."""
    path = os.path.join(tempfile.gettempdir(), f"jutsu_{start_hz}_{end_hz}.wav")
    if not os.path.exists(path):
        t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
        freq = np.linspace(start_hz, end_hz, t.size)
        tone = np.sin(2 * np.pi * np.cumsum(freq) / rate)
        noise = np.random.uniform(-1, 1, t.size) * 0.25
        samples = (tone + noise) * np.linspace(1, 0, t.size) * 0.5
        with wave.open(path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes((samples * 32767).astype(np.int16).tobytes())
    return path


def play_sound(start_hz, end_hz):
    if winsound is None:
        return
    winsound.PlaySound(_sweep_wav(start_hz, end_hz), winsound.SND_FILENAME | winsound.SND_ASYNC)
