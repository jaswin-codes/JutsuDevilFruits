# battle.py - AR monster battle: monsters walk in from both screen sides toward the player.
# Pure game logic (no camera, no drawing) so it can be unit-tested. Positions are frame pixels.
import random

PLAYER_HP = 100
ZONE = 0.12            # player zone = centre band, half-width as a fraction of frame width
CLONE_LINE = 0.25      # while clones are out, monsters are held this far from the centre
MONSTER_RADIUS = 0.05  # fraction of frame width
CONTACT_DAMAGE = 15    # HP a monster takes from you when it reaches the player zone
FIREBALL_DAMAGE = 120
MELEE_DPS = 120        # Chidori; Rasengan's damage is scaled by its size
COOLDOWN = {"fireball": 1.0, "chidori": 5.0, "rasengan": 10.0, "clone": 12.0}
CLONE_SECONDS = 3.0
WAVE_BREAK = 3.0       # seconds between waves
MAX_DT = 0.1           # model loads can stall a frame for >1 s; don't let monsters teleport


class Monster:
    def __init__(self, x, y, side, hp, speed):
        self.x, self.y, self.side = x, y, side   # side -1 = came from the left, 1 = right
        self.hp = self.max_hp = hp
        self.speed = speed
        self.hit_time = -1.0                     # battle time of the last hit (hit flash)


class Battle:
    def __init__(self, frame_w, frame_h, rng=None):
        self.w, self.h = frame_w, frame_h
        self.rng = rng or random.Random()
        self.t = 0.0
        self.hp = PLAYER_HP
        self.score = 0
        self.wave = 0
        self.monsters = []
        self.ready_at = {kind: 0.0 for kind in COOLDOWN}
        self.clone_until = -1.0
        self.hurt_time = -1.0         # battle time a monster last reached you (red flash)
        self.fireball_spent = False   # a cast hits one monster once
        self._start_wave()

    @property
    def radius(self):
        return MONSTER_RADIUS * self.w

    def over(self):
        return self.hp <= 0

    def cooldown_left(self, kind):
        return max(0.0, self.ready_at[kind] - self.t)

    def _start_wave(self):
        self.wave += 1
        self.to_spawn = 2 + self.wave
        self.next_spawn = self.t + (WAVE_BREAK if self.wave > 1 else 1.0)

    def _spawn(self):
        side = self.rng.choice((-1, 1))
        x = -self.radius if side == -1 else self.w + self.radius
        y = self.rng.uniform(0.55, 0.75) * self.h      # low enough for a palm to reach
        # Wave 1 takes ~9 s to reach you: a 6-sign Fireball takes ~4 s to perform
        speed = self.w * (0.04 + 0.006 * (self.wave - 1))
        self.monsters.append(Monster(x, y, side, 100 + 20 * (self.wave - 1), speed))
        self.to_spawn -= 1
        self.next_spawn = self.t + max(1.5, 4.0 - 0.3 * self.wave)

    def on_combo(self, kind):
        """A jutsu was performed. Returns False (and does nothing) while it is cooling down."""
        if self.over() or self.t < self.ready_at[kind]:
            return False
        self.ready_at[kind] = self.t + COOLDOWN[kind]
        if kind == "clone":
            self.clone_until = self.t + CLONE_SECONDS
        if kind == "fireball":
            self.fireball_spent = False
        return True

    def update(self, dt):
        """Advance time: spawn, move monsters, hurt the player. Returns the number of hits taken."""
        if self.over():
            return 0
        dt = min(dt, MAX_DT)
        self.t += dt
        if self.to_spawn and self.t >= self.next_spawn:
            self._spawn()
        centre, hits = self.w / 2, 0
        stop = (CLONE_LINE if self.t < self.clone_until else 0) * self.w
        for m in list(self.monsters):
            # side -1 walks right toward the centre, side 1 walks left
            m.x = min(m.x + m.speed * dt, centre - stop) if m.side == -1 else max(m.x - m.speed * dt, centre + stop)
            if abs(m.x - centre) < ZONE * self.w:
                self.monsters.remove(m)
                self.hp = max(0, self.hp - CONTACT_DAMAGE)
                self.hurt_time = self.t
                hits += 1
        if not self.monsters and not self.to_spawn and not self.over():
            self.score += 50 * self.wave
            self._start_wave()
        return hits

    def _damage(self, m, amount):
        m.hp -= amount
        m.hit_time = self.t
        if m.hp <= 0:
            self.monsters.remove(m)
            self.score += 10 * self.wave
            return True
        return False

    def melee(self, centre, radius, dt, power=1.0):
        """Chidori/Rasengan held at centre: damage every monster the ball overlaps.
        Returns the positions of monsters killed (for particle bursts)."""
        dt, killed = min(dt, MAX_DT), []
        for m in list(self.monsters):
            if (m.x - centre[0]) ** 2 + (m.y - centre[1]) ** 2 < (radius + self.radius) ** 2:
                if self._damage(m, MELEE_DPS * power * dt):
                    killed.append((m.x, m.y))
        return killed

    def fireball(self, mouth_x, centre_x, radius, direction):
        """Launched fireball: hits the nearest monster in the flight direction (any side when
        facing the camera) once its flames reach it in x. Only x is tested: the fireball flies
        at mouth height, above the monsters. Returns killed positions."""
        if self.fireball_spent:
            return []
        ahead = [m for m in self.monsters if direction == 0 or (m.x - mouth_x) * direction > 0]
        if not ahead:
            return []
        target = min(ahead, key=lambda m: abs(m.x - mouth_x))
        if abs(target.x - centre_x) > radius + self.radius:
            return []
        self.fireball_spent = True
        return [(target.x, target.y)] if self._damage(target, FIREBALL_DAMAGE) else []
