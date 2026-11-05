# Webcam-free tests for the AR monster battle (battle.py). Run: python -m pytest
import random

import battle
from battle import Battle, Monster

W, H = 640, 480


def make(*xs):
    """Battle with no pending spawns and monsters at the given x positions."""
    b = Battle(W, H, random.Random(1))
    b.to_spawn = 0
    b.monsters = [Monster(x, 0.65 * H, -1 if x < W / 2 else 1, 100, 30) for x in xs]
    return b


def run(b, seconds, dt=1 / 30):
    hits = 0
    for _ in range(round(seconds / dt)):
        hits += b.update(dt)
    return hits


def test_monsters_spawn_from_both_sides():
    b = Battle(W, H, random.Random(0))
    sides = set()
    for _ in range(20):
        b._spawn()
        sides.add(b.monsters[-1].side)
    assert sides == {-1, 1}
    assert all(m.x < 0 or m.x > W for m in b.monsters)


def test_first_wave_gives_time_for_a_fireball():
    b = Battle(W, H, random.Random(0))
    b._spawn()
    m = b.monsters[0]
    assert (abs(m.x - W / 2) - battle.ZONE * W) / m.speed > 8


def test_reaching_the_player_costs_hp_and_removes_monster():
    b = make(W / 2 - battle.ZONE * W - 5)
    assert run(b, 1) == 1
    assert b.hp == battle.PLAYER_HP - battle.CONTACT_DAMAGE
    assert not b.monsters


def test_fireball_hits_only_nearest_monster_ahead_once():
    b = make(500, 600, 100)
    mouth = W / 2
    assert b.fireball(mouth, 400, 30, 1) == []                # flames not there yet
    assert b.fireball(mouth, 480, 30, 1) == [(500, 0.65 * H)]  # nearest on the right dies
    assert b.fireball(mouth, 600, 30, 1) == []                # spent: one hit per cast
    assert [m.x for m in b.monsters] == [600, 100]
    b.on_combo("fireball")
    assert b.fireball(mouth, mouth, 200, 0) == [(100, 0.65 * H)]  # facing the camera: nearest side


def test_chidori_damage_accumulates_with_time():
    b = make(200)
    m = b.monsters[0]
    for _ in range(15):
        b.melee((200, m.y), 20, 1 / 30)
    assert m.hp == 100 - battle.MELEE_DPS * 0.5
    b.melee((10, 10), 20, 1 / 30)                              # far away: no damage
    assert m.hp == 100 - battle.MELEE_DPS * 0.5


def test_rasengan_damage_scales_with_power():
    small, big = make(200), make(200)
    y = small.monsters[0].y
    small.melee((200, y), 20, 0.1, power=0.5)
    big.melee((200, y), 20, 0.1, power=2.0)
    assert 100 - big.monsters[0].hp == 4 * (100 - small.monsters[0].hp)


def test_melee_kill_scores_and_reports_position():
    b = make(200)
    y = b.monsters[0].y
    killed = []
    for _ in range(20):
        killed += b.melee((200, y), 20, 0.1)
    assert killed == [(200, y)] and b.score == 10


def test_clone_blocks_for_three_seconds_then_damage_resumes():
    b = make(W / 2 - battle.ZONE * W - 5)
    assert b.on_combo("clone")
    assert run(b, 2.9) == 0 and b.hp == battle.PLAYER_HP
    assert b.monsters[0].x == W / 2 - battle.CLONE_LINE * W    # held (pushed back) at the clones
    assert run(b, 10) == 1


def test_cooldown_rejects_a_second_cast():
    b = make(100)
    assert b.on_combo("chidori")
    assert not b.on_combo("chidori")
    run(b, battle.COOLDOWN["chidori"] + 0.1)
    assert b.on_combo("chidori")


def test_wave_advances_when_cleared():
    b = make(200)
    b.melee((200, b.monsters[0].y), 20, 0.1, power=10)
    b.update(0.01)
    assert b.wave == 2 and b.to_spawn == 4 and b.score == 10 + 50


def test_game_over_at_zero_hp_stops_everything():
    b = make()
    b.hp = 1
    b.monsters = [Monster(W / 2 - battle.ZONE * W - 1, 300, -1, 100, 30)]
    b.update(0.1)
    assert b.over() and b.hp == 0
    assert not b.on_combo("fireball")
    assert b.update(1) == 0


def test_huge_dt_is_clamped():
    b = make(50)
    b.update(5.0)
    assert b.monsters[0].x == 50 + 30 * battle.MAX_DT
