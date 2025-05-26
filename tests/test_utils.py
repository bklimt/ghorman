
from numpy import float64

from ghorman.utils import cross, equal, magnitude, matrix, multiply, normalize, point, vector


def test_point():
    v = point(1, 2, 3)
    assert equal(v[0], float64(1))
    assert equal(v[1], float64(2))
    assert equal(v[2], float64(3))
    assert equal(v[3], float64(1))


def test_vector():
    v = vector(1, 2, 3)
    assert equal(v[0], float64(1))
    assert equal(v[1], float64(2))
    assert equal(v[2], float64(3))
    assert equal(v[3], float64(0))


def test_addition():
    p = point(1, 2, 3)
    v = vector(4, 5, 6)
    s = p + v
    assert equal(s, point(5, 7, 9))


def test_subtracting_vector_from_point():
    p = point(1, 2, 3)
    v = vector(4, 5, 6)
    s = p - v
    assert equal(s, point(-3, -3, -3))


def test_subtracting_tow_vectors():
    v1 = vector(5, 7, 9)
    v2 = vector(4, 5, 6)
    s = v1 - v2
    assert equal(s, vector(1, 2, 3))


def test_negating_vector():
    v = vector(5, 7, 9)
    s = -v
    assert equal(s, vector(-5, -7, -9))


def test_multiplying_vector_by_scalar():
    v = vector(5, 7, 9)
    s = 3.5 * v
    assert equal(s, vector(17.5, 24.5, 31.5))


def test_dividing_vector_by_scalar():
    v = vector(5, 7, 9)
    s = v / 2
    assert equal(s, vector(2.5, 3.5, 4.5))


def test_magnitude():
    v = vector(3, 4, 0)
    s = magnitude(v)
    assert equal(s, float64(5))


def test_normalize():
    v = vector(4, 0, 0)
    s = normalize(v)
    assert equal(s, vector(1, 0, 0))

    v = vector(1, 2, 3)
    s = normalize(v)
    assert equal(s, vector(0.26726, 0.53452, 0.80178))


def test_dot_product():
    v1 = vector(1, 2, 3)
    v2 = vector(2, 3, 4)
    assert equal(v1.dot(v2), float64(20))


def test_cross_product():
    v1 = vector(1, 2, 3)
    v2 = vector(2, 3, 4)
    assert equal(cross(v1, v2), vector(-1, 2, -1))
    assert equal(cross(v2, v1), vector(1, -2, 1))


def test_matrix_multiplication():
    a = matrix([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 8, 7, 6],
        [5, 4, 3, 2],
    ])
    b = matrix([
        [-2, 1, 2, 3],
        [3, 2, 1, -1],
        [4, 3, 6, 5],
        [1, 2, 7, 8],
    ])
    c = multiply(a, b)

    expected = matrix([
        [20, 22, 50, 48],
        [44, 54, 114, 108],
        [40, 58, 110, 102],
        [16, 26, 46, 42]
    ])

    assert equal(c, expected)
