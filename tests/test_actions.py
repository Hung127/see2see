"""Tests for action decoding (DESIGN §5.3)."""
import numpy as np

from evolve.actions import Actions, decode


def outputs(**overrides):
    base = np.full((1, 8), 0.25)
    base[0, 0] = 0.5
    base[0, 1] = 0.25
    base[0, 2] = 0.8
    for k, v in overrides.items():
        base[0, int(k[1:])] = v
    return base


def phen(n=1, size=1.0):
    p = np.zeros((n, 12))
    p[:, 0] = size
    return p


def test_decode_pure_function():
    o = outputs()
    a1 = decode(o, phen())
    a2 = decode(o.copy(), phen())
    for name in ("throttle", "turn", "eat", "attack", "sprint", "reproduce", "rest"):
        assert np.array_equal(getattr(a1, name), getattr(a2, name)), name


def test_escape_does_not_steer():
    o1 = outputs(o5=0.1)
    o2 = outputs(o5=0.9)
    a1 = decode(o1, phen())
    a2 = decode(o2, phen())
    assert np.array_equal(a1.turn, a2.turn)  # escape may never change heading
    assert a2.sprint[0] > a1.sprint[0]       # but it does scale speed


def test_rest_blocks_feeding():
    a = decode(outputs(o7=0.9, o3=0.9, o4=0.9), phen())
    assert not a.eat[0]
    assert not a.attack[0]
    assert a.rest[0]
    assert a.throttle[0] < 0.5 * 0.8 * 0.5  # throttle scaled by (1 - o7)


def test_thresholds():
    below = decode(outputs(o3=0.499), phen())
    above = decode(outputs(o3=0.501), phen())
    assert not below.eat[0]
    assert above.eat[0]
    below_a = decode(outputs(o4=0.499), phen())
    above_a = decode(outputs(o4=0.501), phen())
    assert not below_a.attack[0]
    assert above_a.attack[0]
    below_r = decode(outputs(o6=0.499), phen())
    above_r = decode(outputs(o6=0.501), phen())
    assert not below_r.reproduce[0]
    assert above_r.reproduce[0]


def test_throttle_and_turn_formulas():
    o = outputs(o0=0.5, o2=0.8, o1=0.25, o7=0.0, o5=0.0)
    a = decode(o, phen(size=1.0))
    assert a.throttle[0] == 0.5 * 0.8          # |o0| * o2
    assert a.turn[0] == 0.25 * 3.0 / 1.0 ** 0.33  # o1 * turn_rate_max
    assert a.sprint[0] == 1.0


def test_negative_forward_gives_positive_throttle():
    o = outputs(o0=-0.5, o2=0.8, o7=0.0)
    a = decode(o, phen())
    assert a.throttle[0] == 0.5 * 0.8


def test_actions_is_dataclass_of_arrays():
    a = decode(np.zeros((3, 8)), phen(3))
    assert isinstance(a, Actions)
    for name in ("throttle", "turn", "eat", "attack", "sprint", "reproduce", "rest"):
        assert getattr(a, name).shape == (3,), name
