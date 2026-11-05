# Webcam-free tests for the two-player duel (duel.py). Run: python -m pytest
import duel
from battle import PLAYER_HP
from duel import Duel

W, H = 640, 480
DT = 1 / 30


def tick(d, seconds):
    for _ in range(round(seconds / DT)):
        d.update(DT)


def test_cooldowns_are_per_player():
    d = Duel(W, H)
    p1, p2 = d.fighters
    assert p1.on_combo("chidori")
    assert not p1.on_combo("chidori")         # cooling down
    assert p2.on_combo("chidori")             # the other player's cooldown is separate
    tick(d, 5.1)
    assert p1.on_combo("chidori")


def test_fireball_hits_opponent_once_per_cast():
    d = Duel(W, H)
    p1, p2 = d.fighters
    p1.on_combo("fireball")
    assert not d.fireball(0, d.w * 0.4, 20)   # still in flight, short of the opponent
    assert d.fireball(0, d.body_x(1), 20)
    assert not d.fireball(0, d.body_x(1), 20) # same cast can't hit twice
    assert p2.hp == PLAYER_HP - duel.FIREBALL_DAMAGE
    assert p1.hp == PLAYER_HP


def test_fireball_from_the_right_player_hits_the_left():
    d = Duel(W, H)
    d.fighters[1].on_combo("fireball")
    assert d.fireball(1, d.body_x(0), 20)
    assert d.fighters[0].hp < PLAYER_HP


def test_shadow_clone_blocks_and_uses_up_the_fireball():
    d = Duel(W, H)
    p1, p2 = d.fighters
    p2.on_combo("clone")
    p1.on_combo("fireball")
    assert not d.fireball(0, d.body_x(1), 20)
    assert p2.hp == PLAYER_HP and p2.blocked_time == d.t
    tick(d, duel.SHIELD_SECONDS + 0.1)
    assert not d.fireball(0, d.body_x(1), 20)  # blocked cast is spent, can't land later
    assert p2.hp == PLAYER_HP


def test_shadow_clone_blocks_melee_until_it_ends():
    d = Duel(W, H)
    p1, p2 = d.fighters
    p2.on_combo("clone")
    p1.on_combo("chidori")
    reach = (W / 2 + 30, H / 2)
    assert not d.melee(0, reach, 20, DT)
    tick(d, duel.SHIELD_SECONDS + 0.1)
    assert d.melee(0, reach, 20, DT)
    assert p2.hp < PLAYER_HP


def test_invisibility_dodges_fireball_but_not_melee():
    d = Duel(W, H)
    p1, p2 = d.fighters
    p2.on_combo("invisible")
    p1.on_combo("fireball")
    assert not d.fireball(0, d.body_x(1), 20)
    assert not p1.fireball_spent               # flies on: lands if they reappear in its path
    p1.on_combo("chidori")
    assert d.melee(0, (W / 2 + 30, H / 2), 20, DT)
    tick(d, duel.HIDDEN_SECONDS + 0.1)
    assert d.fireball(0, d.body_x(1), 20)


def test_melee_hits_only_when_the_ball_crosses_the_centre_line():
    d = Duel(W, H)
    d.fighters[0].on_combo("chidori")
    d.fighters[1].on_combo("rasengan")
    assert not d.melee(0, (W * 0.3, H / 2), 40, DT)        # held in front of yourself
    assert d.melee(0, (W / 2 - 10, H / 2), 40, DT)         # palm on own side, ball pokes across
    assert not d.melee(1, (W * 0.7, H / 2), 40, DT)
    assert d.melee(1, (W / 2 + 10, H / 2), 40, DT)


def test_no_single_cast_wins_from_full_hp():
    d = Duel(W, H)
    p1, p2 = d.fighters
    p1.on_combo("rasengan")
    for _ in range(300):                       # 10 s of the strongest Rasengan held across
        d.melee(0, (W / 2, H / 2), 200, DT, power=2.5)
        d.update(DT)
    assert abs(p2.hp - (PLAYER_HP - duel.MELEE_MAX)) < 1e-6
    p1.on_combo("fireball")
    d.fireball(0, d.body_x(1), 20)
    assert p2.hp > 0
    assert duel.FIREBALL_DAMAGE < PLAYER_HP and duel.MELEE_MAX < PLAYER_HP


def test_knockout_ends_the_round():
    d = Duel(W, H)
    p1, p2 = d.fighters
    p2.hp = 10
    p1.on_combo("fireball")
    assert d.fireball(0, d.body_x(1), 20)
    assert d.over() and d.winner() == 0 and p2.hp == 0
    assert not p1.on_combo("chidori") and not p2.on_combo("chidori")
    t = d.t
    d.update(DT)
    assert d.t == t                            # clock stops on the winner screen


def test_same_frame_knockouts_first_hit_wins():
    d = Duel(W, H)
    p1, p2 = d.fighters
    p1.hp = p2.hp = 0.1
    p1.on_combo("chidori")
    p2.on_combo("chidori")
    assert d.melee(0, (W / 2, H / 2), 40, DT)
    assert d.over() and d.winner() == 0
    assert not d.melee(1, (W / 2, H / 2), 40, DT)   # over: no hits after the KO
    p1.hp = 0
    assert d.winner() is None                  # both down would be a draw


def test_long_frame_is_clamped():
    d = Duel(W, H)
    d.update(5.0)                              # model load stalls a frame
    assert d.t == duel.MAX_DT
