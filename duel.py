# duel.py - two-player jutsu duel: two people side by side in front of one webcam, the screen
# split down the middle (player 0 = left half, player 1 = right half). Pure rules (no camera,
# no drawing) so they can be unit-tested. Positions are frame pixels.
from battle import COOLDOWN, MAX_DT, PLAYER_HP, SUB_SECONDS

BODY = 0.12            # fireball target: centre of the opponent's half +- this fraction of frame width
FRONT = -0.03          # Chidori/Rasengan hit once the ball's edge gets this close to the centre line
                       # (negative: it needn't cross; hands are ~30 px wide two metres from a webcam)
FIREBALL_DAMAGE = 40   # the six-sign Fireball is the slowest jutsu to perform, so it hits hardest
MELEE_DPS = 20         # Chidori; Rasengan's damage is scaled by its size
MELEE_MAX = 35         # most one Chidori/Rasengan cast can deal: no single jutsu wins a round
SHIELD_SECONDS = 3.0   # Shadow Clone: the clones take every hit
HIDDEN_SECONDS = 5.0   # invisible: the opponent's fireball can't find you
SUMMON_DAMAGE = 30     # five signs plus a slap, so it hits hard, but still can't win a round alone
SHARINGAN_DELAY = 3.0  # Sharingan: every opponent jutsu becomes ready this much later
COUNTDOWN = 3.0        # 3-2-1 before each round; jutsu do nothing until FIGHT
ROUND_BREAK = 3.0      # round-winner screen before the next round starts
ROUNDS_TO_WIN = 2      # best of 3


class Fighter:
    """One player's HP, cooldowns and defences. Same on_combo/cooldown_left/over API as
    battle.Battle, so main.py gates combos the same way in both modes."""

    def __init__(self, duel):
        self.duel = duel                 # shared clock: cooldowns and shields of both players agree
        self.hp = PLAYER_HP
        self.ready_at = {kind: 0.0 for kind in COOLDOWN}
        self.shield_until = self.hidden_until = self.sub_until = -1.0
        self.swap_time = -9.0            # duel time a log took a hit in your place
        self.hurt_time = self.blocked_time = -1.0   # duel time of the last hit taken / blocked
        self.fireball_spent = True       # a cast hits once
        self.melee_dealt = 0.0           # damage dealt by the current Chidori/Rasengan cast

    def over(self):
        """True while jutsu can't be used: countdown, between rounds, match over."""
        return not self.duel.fighting()

    def cooldown_left(self, kind):
        return max(0.0, self.ready_at[kind] - self.duel.t)

    def shielded(self):
        return self.duel.t < self.shield_until

    def hidden(self):
        return self.duel.t < self.hidden_until

    def swap_armed(self):
        return self.duel.t < self.sub_until

    def on_combo(self, kind):
        """A jutsu was performed. Returns False (and does nothing) while it cools down or nobody may fight."""
        t = self.duel.t
        if self.over() or t < self.ready_at[kind]:
            return False
        self.ready_at[kind] = t + COOLDOWN[kind]
        if kind == "clone":
            self.shield_until = t + SHIELD_SECONDS
        elif kind == "invisible":
            self.hidden_until = t + HIDDEN_SECONDS
        elif kind == "substitution":
            self.sub_until = t + SUB_SECONDS
        elif kind == "fireball":
            self.fireball_spent = False
        elif kind == "sharingan":       # it reads the opponent: their cooldowns, ready or not, run longer
            foe = next(f for f in self.duel.fighters if f is not self)
            foe.ready_at = {k: max(ready, t) + SHARINGAN_DELAY for k, ready in foe.ready_at.items()}
        elif kind in ("chidori", "rasengan"):
            self.melee_dealt = 0.0
        return True


class Duel:
    """A best-of-3 match. self.fighters is replaced at each new round (full HP, no cooldowns)."""

    def __init__(self, frame_w, frame_h):
        self.w, self.h = frame_w, frame_h
        self.t = 0.0
        self.round = 0
        self.wins = [0, 0]
        self._next_round()

    def _next_round(self):
        self.round += 1
        self.fighters = [Fighter(self), Fighter(self)]
        self.fight_at = self.t + COUNTDOWN
        self.ko_at = None
        self.round_winner = None

    def body_x(self, i):
        return self.w * (0.25 + 0.5 * i)

    def toward(self, i):
        """Screen direction from player i to the opponent: 1 = right, -1 = left."""
        return 1 if i == 0 else -1

    def over(self):
        """This round has a knockout."""
        return self.ko_at is not None

    def fighting(self):
        return not self.over() and self.t >= self.fight_at

    def winner(self):
        """Player who won this round (0 or 1), None while it is still on."""
        return self.round_winner

    def match_winner(self):
        """Player who won the match, None until someone has ROUNDS_TO_WIN rounds."""
        return next((i for i, n in enumerate(self.wins) if n >= ROUNDS_TO_WIN), None)

    def update(self, dt):
        if self.match_winner() is not None:
            return                      # winner screen until R
        self.t += min(dt, MAX_DT)
        if self.over() and self.t - self.ko_at >= ROUND_BREAK:
            self._next_round()

    def _hit(self, i, amount):
        """Player i's attack lands on the opponent unless their clones or a substitution log take it."""
        foe = self.fighters[1 - i]
        if foe.shielded():
            foe.blocked_time = self.t
            return False
        if foe.swap_armed():             # one hit lands on the log, then the swap is used up
            foe.sub_until, foe.swap_time = -1.0, self.t
            return False
        foe.hp = max(0.0, foe.hp - amount)
        foe.hurt_time = self.t
        if foe.hp == 0:
            self.ko_at, self.round_winner = self.t, i
            self.wins[i] += 1
        return True

    def melee(self, i, centre, radius, dt, power=1.0):
        """Chidori/Rasengan held by player i at centre: damages the opponent while the ball
        reaches the centre line. The palm itself stays on its own side, so main.py can
        assign hands by screen half. Returns True on a hit."""
        me = self.fighters[i]
        reach = (centre[0] - self.w / 2) * self.toward(i) + radius   # ball edge past the line (< 0: short)
        if not self.fighting() or reach < FRONT * self.w or me.melee_dealt >= MELEE_MAX:
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
        if not self.fighting() or me.fireball_spent or foe.hidden():
            return False
        if abs(centre_x - self.body_x(1 - i)) > BODY * self.w + radius:
            return False
        me.fireball_spent = True
        return self._hit(i, FIREBALL_DAMAGE)

    def summon(self, i):
        """Player i's summoned creature charges the opponent: one hit, blocked like any other.
        Returns True on a hit."""
        return self.fighting() and self._hit(i, SUMMON_DAMAGE)
