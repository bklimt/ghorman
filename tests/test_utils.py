
from ghorman.utils import equal, point, vector


def test_point():
    v = point(1, 2, 3)
    assert equal(v[0], 1)
    assert equal(v[1], 2)
    assert equal(v[2], 3)
    assert equal(v[3], 1)


def test_vector():
    v = vector(1, 2, 3)
    assert equal(v[0], 1)
    assert equal(v[1], 2)
    assert equal(v[2], 3)
    assert equal(v[3], 0)


def test_addition():
    p = point(1, 2, 3)
    v = vector(4, 5, 6)
    s = p + v
    assert equal(s[0], 5)
    assert equal(s[1], 7)
    assert equal(s[2], 9)
    assert equal(s[3], 1)
