
from numpy import allclose, array, float64, isclose, ndarray, sqrt
from numpy.typing import NDArray
from typing import Tuple, overload

EPSILON = 0.00001


def point(x: float, y: float, z: float) -> NDArray[float64]:
    return array([x, y, z, 1.0], dtype=float64)


def vector(x: float, y: float, z: float) -> NDArray[float64]:
    return array([x, y, z, 0.0], dtype=float64)


@overload
def equal(x: float64, y: float64) -> bool:
    ...


@overload
def equal(x: NDArray[float64], y: NDArray[float64]) -> bool:
    ...


def equal(x, y) -> bool:
    if isinstance(x, ndarray):
        return allclose(x, y, atol=EPSILON)
    else:
        return isclose(x, y, atol=EPSILON)


def magnitude(v) -> float64:
    return sqrt(v.dot(v))


def normalize(v) -> NDArray[float64]:
    return v / magnitude(v)
