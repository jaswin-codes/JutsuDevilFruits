# game.py - timed challenge: perform the target jutsu as many times as you can
import random


class Game:
    def __init__(self, combos, now, duration=60.0):
        """combos: the COMBOS table (sign tuple -> combo dict)."""
        self.combos = combos
        self.end = now + duration
        self.score = 0
        self.target = random.choice(list(combos))   # sign tuple to perform

    def time_left(self, now):
        return max(0.0, self.end - now)

    def pause(self, seconds, now):
        """The clock stood still for seconds (the sign chart was open). A finished game stays finished."""
        if not self.over(now):
            self.end += seconds

    def over(self, now):
        return self.time_left(now) == 0

    def on_combo(self, combo, now):
        """Call when a combo fires. Returns True if it was the target (and scores)."""
        if self.over(now) or combo is not self.combos[self.target]:
            return False
        self.score += 1
        others = [signs for signs in self.combos if signs != self.target]
        self.target = random.choice(others or [self.target])
        return True
