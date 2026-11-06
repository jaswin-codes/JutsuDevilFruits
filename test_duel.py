# Webcam-free tests for the two-player duel (duel.py). Run: python -m pytest
import duel
from battle import PLAYER_HP
from duel import Duel

W, H = 640, 480
DT = 1 / 30


def tick(d, seconds):
    for _ in range(round(seconds / DT) + 1):   # +1: float steps fall just short
        d.update(DT)


def ready():
    """Duel with the first round's countdown over."""
    d = Duel(W, H)
    tick(d, duel.COUNTDOWN)
    return d


def test_cooldowns_are_per_player():
    d = ready()
    p1, p2 = d.fighters
    assert p1.on_combo("chidori")
    assert not p1.on_combo("chidori")         # cooling down
    assert p2.on_combo("chidori")             # the other player's cooldown is separate
    tick(d, 5.1)
    assert p1.on_combo("chidori")


def test_fireball_hits_opponent_once_per_cast():
    d = ready()
    p1, p2 = d.fighters
    p1.on_combo("fireball")
    assert not d.fireball(0, d.w * 0.4, 20)   # still in flight, short of the opponent
    assert d.fireball(0, d.body_x(1), 20)
    assert not d.fireball(0, d.body_x(1), 20) # same cast can't hit twice
    assert p2.hp == PLAYER_HP - duel.FIREBALL_DAMAGE
    assert p1.hp == PLAYER_HP


def test_fireball_from_the_right_player_hits_the_left():
    d = ready()
    d.fighters[1].on_combo("fireball")
    assert d.fireball(1, d.body_x(0), 20)
    assert d.fighters[0].hp < PLAYER_HP


def test_shadow_clone_blocks_and_uses_up_the_fireball():
    d = ready()
    p1, p2 = d.fighters
    p2.on_combo("clone")
    p1.on_combo("fireball")
    assert not d.fireball(0, d.body_x(1), 20)
    assert p2.hp == PLAYER_HP and p2.blocked_time == d.t
    tick(d, duel.SHIELD_SECONDS + 0.1)
    assert not d.fireball(0, d.body_x(1), 20)  # blocked cast is spent, can't land later
    assert p2.hp == PLAYER_HP


def test_shadow_clone_blocks_melee_until_it_ends():
    d = ready()
    p1, p2 = d.fighters
    p2.on_combo("clone")
    p1.on_combo("chidori")
    reach = (W / 2 + 30, H / 2)
    assert not d.melee(0, reach, 20, DT)
    tick(d, duel.SHIELD_SECONDS + 0.1)
    assert d.melee(0, reach, 20, DT)
    assert p2.hp < PLAYER_HP


def test_invisibility_dodges_fireball_but_not_melee():
    d = ready()
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
    d = ready()
    d.fighters[0].on_combo("chidori")
    d.fighters[1].on_combo("rasengan")
    assert not d.melee(0, (W * 0.3, H / 2), 40, DT)        # held in front of yourself
    assert d.melee(0, (W / 2 - 50, H / 2), 40, DT)         # palm on own side, ball near the line
    assert not d.melee(1, (W * 0.7, H / 2), 40, DT)
    assert d.melee(1, (W / 2 + 10, H / 2), 40, DT)


def test_no_single_cast_wins_from_full_hp():
    d = ready()
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


def test_no_jutsu_during_the_countdown():
    d = Duel(W, H)
    p1 = d.fighters[0]
    assert not p1.on_combo("fireball") and p1.over()
    tick(d, duel.COUNTDOWN)
    assert p1.on_combo("fireball") and not p1.over()


def knock_out(d, attacker):
    me, foe = d.fighters[attacker], d.fighters[1 - attacker]
    foe.hp = 10
    me.on_combo("fireball")
    assert d.fireball(attacker, d.body_x(1 - attacker), 20)


def test_knockout_ends_the_round_then_next_round_starts_fresh():
    d = ready()
    knock_out(d, 0)
    p1, p2 = d.fighters
    assert d.over() and d.winner() == 0 and p2.hp == 0 and d.wins == [1, 0]
    assert not p1.on_combo("chidori") and not p2.on_combo("chidori")
    assert not d.melee(1, (W / 2, H / 2), 40, DT)   # no hits after the KO
    tick(d, duel.ROUND_BREAK)
    assert d.round == 2 and not d.over() and d.winner() is None
    assert all(f.hp == PLAYER_HP and f.cooldown_left("fireball") == 0 for f in d.fighters)
    assert d.fighters[0].over()                # countdown again
    tick(d, duel.COUNTDOWN)
    assert d.fighting()


def test_best_of_three():
    d = ready()
    knock_out(d, 0)
    tick(d, duel.ROUND_BREAK)
    tick(d, duel.COUNTDOWN)
    knock_out(d, 1)
    assert d.match_winner() is None
    tick(d, duel.ROUND_BREAK)
    tick(d, duel.COUNTDOWN)
    assert d.round == 3
    knock_out(d, 1)
    assert d.match_winner() == 1 and d.wins == [1, 2]
    t = d.t
    tick(d, 10)
    assert d.t == t and d.round == 3           # winner screen stays until R


def test_long_frame_is_clamped():
    d = Duel(W, H)
    d.update(5.0)                              # model load stalls a frame
    assert d.t == duel.MAX_DT


def test_duel_view_draws_hud_and_winner():
    import numpy as np
    import main  # noqa: F401  - main.py must at least import with --duel wired in
    from duel_view import draw_duel
    d = ready()
    frame = np.zeros((H, W, 3), np.uint8)
    draw_duel(frame, d, ("KATON - FIREBALL JUTSU - cooling down 1s", None))
    knock_out(d, 0)
    draw_duel(frame, d)
    assert frame.any()


def test_substitution_takes_one_hit_then_is_used_up():
    d = ready()
    p1, p2 = d.fighters
    p2.on_combo("substitution")
    p1.on_combo("fireball")
    assert not d.fireball(0, d.body_x(1), 20)    # the log took it
    assert p2.hp == PLAYER_HP and p2.swap_time == d.t and not p2.swap_armed()
    p1.on_combo("chidori")
    assert d.melee(0, (W / 2, H / 2), 40, DT)    # next hit lands


def test_draw_log_draws_then_vanishes():
    import numpy as np
    from anime_effects import draw_log
    frame = np.zeros((H, W, 3), np.uint8)
    draw_log(frame, (320, 300), 120, 0.1)
    assert frame.any()
    frame[:] = 0
    draw_log(frame, (320, 300), 120, 5.0)        # after its duration: nothing
    draw_log(frame, (320, 300), 120, -1.0)       # never swapped: nothing
    assert not frame.any()


def test_summon_hits_once_and_clones_block_it():
    d = ready()
    p1, p2 = d.fighters
    assert d.summon(0) and p2.hp == PLAYER_HP - duel.SUMMON_DAMAGE
    p1.on_combo("clone")
    assert not d.summon(1) and p1.hp == PLAYER_HP


def test_sharingan_delays_every_opponent_cooldown():
    d = ready()
    p1, p2 = d.fighters
    p2.on_combo("chidori")
    chidori_ready = p2.ready_at["chidori"]
    assert p1.on_combo("sharingan")
    assert p2.ready_at["chidori"] == chidori_ready + duel.SHARINGAN_DELAY
    assert p2.cooldown_left("fireball") == duel.SHARINGAN_DELAY     # even a ready jutsu waits
    assert not p2.on_combo("fireball") and p1.cooldown_left("fireball") == 0
