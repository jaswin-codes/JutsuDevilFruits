import numpy as np
from effects import ParticleSystem
from mera import HIKEN_TIME, Mera, burn, knuckles
from test_one_piece import OPEN, fist


def scaled(lm, k):
    """The hand k times bigger on screen (nearer the camera), around its wrist."""
    x0, y0 = lm[0]
    return [(x0 + (x - x0) * k, y0 + (y - y0) * k) for x, y in lm]


def run(mera, frames, dt=0.05):
    return [mera.update(hands, dt) for hands in frames]


def test_knuckle_width_grows_with_the_hand():
    assert abs(knuckles(scaled(fist(), 1.5)) - 1.5 * knuckles(fist())) < 1e-6


def test_fast_punch_fires_one_hiken():
    mera = Mera()
    fired = run(mera, [[fist()], [scaled(fist(), 1.2)], [scaled(fist(), 1.5)]] + [[scaled(fist(), 1.5)]] * 10)
    assert fired.count(True) == 1 and mera.blasting()
    assert mera.fists and mera.fists[0][1] > 60


def test_slow_approach_and_open_hands_dont_fire():
    mera = Mera()
    assert not any(run(mera, [[scaled(fist(), 1 + 0.05 * i)] for i in range(12)], dt=0.1))
    assert not any(run(Mera(), [[OPEN], [scaled(OPEN, 1.6)]]))


def test_a_second_punch_needs_the_blast_over_and_a_new_punch():
    mera = Mera()
    run(mera, [[fist()], [scaled(fist(), 1.5)]])
    assert not any(run(mera, [[fist()], [scaled(fist(), 1.5)]]))       # still blasting
    run(mera, [[]] * int(HIKEN_TIME / 0.05 + 2))
    assert not mera.blasting()
    assert any(run(mera, [[fist()], [scaled(fist(), 1.5)]]))


def test_switching_to_a_bigger_fist_elsewhere_is_not_a_punch():
    small = [(x - 200, y) for x, y in fist()]
    big = [(x + 200, y) for x, y in scaled(fist(), 1.5)]
    assert not any(run(Mera(), [[small], [big]]))


def test_burning_fist_sends_flames_up():
    particles = ParticleSystem()
    burn(particles, (300, 300), 40)
    assert len(particles.life) == 24 and particles.vel[:, 1].mean() < -50


def test_hiken_draws_fire_and_survives_the_frame_edge():
    mera = Mera()
    run(mera, [[fist()], [scaled(fist(), 1.5)]] + [[]] * 12)            # 0.6 s in: launched
    frame = np.zeros((480, 640, 3), np.uint8)
    mera.draw_hiken(frame)
    assert frame[int(mera.hiken_from[1]), int(mera.hiken_from[0])].sum() > 300
    mera.hiken_from = np.array([-50.0, 470.0])
    mera.draw_hiken(frame)
