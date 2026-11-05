# duel.py - two-player jutsu duel: two people side by side in front of one webcam, the screen
# split down the middle (player 0 = left half, player 1 = right half). Pure rules (no camera,
# no drawing) so they can be unit-tested. Positions are frame pixels.
from battle import COOLDOWN, MAX_DT, PLAYER_HP

BODY = 0.12            # fireball target: centre of the opponent's half +- this fraction of frame width
FRONT = 0.03           # Chidori/Rasengan hit once the ball crosses this far past the centre line
FIREBALL_DAMAGE = 30
MELEE_DPS = 20         # Chidori; Rasengan's damage is scaled by its size
MELEE_MAX = 35         # most one Chidori/Rasengan cast can deal: no single jutsu wins a round
SHIELD_SECONDS = 3.0   # Shadow Clone: the clones take every hit
HIDDEN_SECONDS = 5.0   # invisible: the opponent's fireball can't find you


class Fighter:
    """One player's HP, cooldowns and defences. Same on_combo/cooldown_left/over API as
    battle.Battle, so main.py gates combos the same way in both modes."""

    def __init__(self, duel):
        self.duel = duel                 # shared clock: cooldowns and shields of both players agree
        self.hp = PLAYER_HP
        self.ready_at = {kind: 0.0 for kind in COOLDOWN}
        self.shield_until = self.hidden_until = -1.0
        self.hurt_time = self.blocked_time = -1.0   # duel time of the last hit taken / blocked
        self.fireball_spent = True       # a cast hits once
        self.melee_dealt = 0.0           # damage dealt by the current Chidori/Rasengan cast

    def over(self):
        return self.duel.over()

    def cooldown_left(self, kind):
        return max(0.0, self.ready_at[kind] - self.duel.t)

    def shielded(self):
        return self.duel.t < self.shield_until

    def hidden(self):
        return self.duel.t < self.hidden_until

    def on_combo(self, kind):
        """A jutsu was performed. Returns False (and does nothing) while it cools down or the round is over."""
        t = self.duel.t
        if self.over() or t < self.ready_at[kind]:
            return False
        self.ready_at[kind] = t + COOLDOWN[kind]
        if kind == "clone":
            self.shield_until = t + SHIELD_SECONDS
        elif kind == "invisible":
            self.hidden_until = t + HIDDEN_SECONDS
        elif kind == "fireball":
            self.fireball_spent = False
        else:
            self.melee_dealt = 0.0
        return True


class Duel:
    def __init__(self, frame_w, frame_h):
        self.w, self.h = frame_w, frame_h
        self.t = 0.0
        self.fighters = [Fighter(self), Fighter(self)]

    def body_x(self, i):
        return self.w * (0.25 + 0.5 * i)

    def toward(self, i):
        """Screen direction from player i to the opponent: 1 = right, -1 = left."""
        return 1 if i == 0 else -1

    def over(self):
        return any(f.hp <= 0 for f in self.fighters)

    def winner(self):
        """0 or 1, None while fighting or after a double KO."""
        alive = [i for i, f in enumerate(self.fighters) if f.hp > 0]
        return alive[0] if self.over() and len(alive) == 1 else None

    def update(self, dt):
        if not self.over():
            self.t += min(dt, MAX_DT)

    def _hit(self, i, amount):
        """Player i's attack lands on the opponent unless their clones take it."""
        foe = self.fighters[1 - i]
        if foe.shielded():
            foe.blocked_time = self.t
            return False
        foe.hp = max(0.0, foe.hp - amount)
        foe.hurt_time = self.t
        return True

    def melee(self, i, centre, radius, dt, power=1.0):
        """Chidori/Rasengan held by player i at centre: damages the opponent while the ball
        reaches across the centre line. The palm itself stays on its own side, so main.py can
        assign hands by screen half. Returns True on a hit."""
        me = self.fighters[i]
        reach = (centre[0] - self.w / 2) * self.toward(i) + radius   # how far the ball pokes over the line
        if self.over() or reach < FRONT * self.w or me.melee_dealt >= MELEE_MAX:
            return False
        amount = min(MELEE_DPS * power * min(dt, MAX_DT), MELEE_MAX - me.melee_dealt)
        if not self._hit(i, amount):
            return False
        me.melee_dealt += amount
        return True

    def fireball(self, i, centre_x, radius):
        """Launched fireball of player i at centre_x: hits once when its flames reach the
        opponent's body band. A shadow-clone block also uses it up; an invisible opponent
        can't be found, so it flies on. Returns True on a hit."""
        me, foe = self.fighters[i], self.fighters[1 - i]
        if self.over() or me.fireball_spent or foe.hidden():
            return False
        if abs(centre_x - self.body_x(1 - i)) > BODY * self.w + radius:
            return False
        me.fireball_spent = True
        return self._hit(i, FIREBALL_DAMAGE)
