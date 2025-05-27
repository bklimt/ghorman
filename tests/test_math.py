
import pytest

from numpy import float64
from numpy.linalg import LinAlgError

from ghorman.math import cofactor, cross, det, equal, identity, inverse, magnitude, matrix, minor, multiply, normalize, point, submatrix, vector


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


def test_matrix_array_multiplication():
    a = matrix([
        [1, 2, 3, 4],
        [2, 4, 4, 2],
        [8, 6, 4, 1],
        [0, 0, 0, 1],
    ])
    b = point(1, 2, 3)
    c = multiply(a, b)

    expected = point(18, 24, 33)

    assert equal(c, expected)


def test_identity_matrix():
    a = matrix([
        [0, 1, 2, 4],
        [1, 2, 4, 8],
        [2, 4, 8, 16],
        [4, 8, 16, 32],
    ])
    b = multiply(a, identity())
    assert equal(b, a)

    a = point(1, 2, 3)
    b = multiply(a, identity())
    assert equal(b, a)


def test_transpose():
    a = matrix([
        [0, 9, 3, 0],
        [9, 8, 0, 8],
        [1, 8, 5, 3],
        [0, 0, 5, 8],
    ])
    b = a.transpose()

    expected = matrix([
        [0, 9, 1, 0],
        [9, 8, 8, 0],
        [3, 0, 5, 5],
        [0, 8, 3, 8],
    ])

    assert equal(b, expected)


def test_deteriminant2x2():
    a = matrix([
        [1, 5],
        [-3, 2],
    ])
    d = det(a)

    assert equal(d, float64(17))


def test_submatrix():
    m = matrix([
        [1, 5, 0],
        [-3, 2, 7],
        [0, 6, -3],
    ])

    assert equal(submatrix(m, 0, 2), matrix([[-3, 2], [0, 6]]))

    m = matrix([
        [-6, 1, 1, 6],
        [-8, 5, 8, 6],
        [-1, 0, 8, 2],
        [-7, 1, -1, 1],
    ])

    assert equal(submatrix(m, 2, 1), matrix([
        [-6, 1, 6],
        [-8, 8, 6],
        [-7, -1, 1],
    ]))


def test_minor():
    a = matrix([
        [3, 5, 0],
        [2, 1, -7],
        [6, -1, 5],
    ])

    assert equal(minor(a, 1, 0), float64(25))


def test_cofactor():
    a = matrix([
        [3, 5, 0],
        [2, -1, -7],
        [6, -1, 5],
    ])
    assert equal(minor(a, 0, 0), float64(-12))
    assert equal(cofactor(a, 0, 0), float64(-12))
    assert equal(minor(a, 1, 0), float64(25))
    assert equal(cofactor(a, 1, 0), float64(-25))


def test_determinant3x3():
    a = matrix([
        [1, 2, 6],
        [-5, 8, -4],
        [2, 6, 4],
    ])
    assert equal(cofactor(a, 0, 0), float64(56))
    assert equal(cofactor(a, 0, 1), float64(12))
    assert equal(cofactor(a, 0, 2), float64(-46))
    assert equal(det(a), float64(-196))

    a = matrix([
        [-2, -8, 3, 5],
        [-3, 1, 7, 3],
        [1, 2, -9, 6],
        [-6, 7, 7, -9],
    ])
    assert equal(cofactor(a, 0, 0), float64(690))
    assert equal(cofactor(a, 0, 1), float64(447))
    assert equal(cofactor(a, 0, 2), float64(210))
    assert equal(cofactor(a, 0, 3), float64(51))
    assert equal(det(a), float64(-4071))


def test_inv():
    a = matrix([
        [6, 4, 4, 4],
        [5, 5, 7, 6],
        [4, -9, 3, -7],
        [9, 1, 7, -6],
    ])
    assert equal(det(a), float64(-2120))
    assert inverse(a) is not None

    a = matrix([
        [-4, 2, -2, -3],
        [9, 6, 2, 6],
        [0, -5, 1, -5],
        [0, 0, 0, 0],
    ])
    assert equal(det(a), float64(0))
    with pytest.raises(LinAlgError):
        inverse(a)
