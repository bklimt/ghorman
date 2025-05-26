
import numpy as np

from numpy.typing import NDArray
from typing import overload

EPSILON = 0.00001


def point(x: float, y: float, z: float) -> NDArray[np.float64]:
    return np.array([x, y, z, 1.0], dtype=np.float64)


def vector(x: float, y: float, z: float) -> NDArray[np.float64]:
    return np.array([x, y, z, 0.0], dtype=np.float64)


@overload
def equal(x: np.float64, y: np.float64) -> bool:
    ...


@overload
def equal(x: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    ...


def equal(x, y) -> bool:
    if isinstance(x, np.ndarray):
        return np.allclose(x, y, atol=EPSILON)
    else:
        return np.isclose(x, y, atol=EPSILON)


def magnitude(v) -> np.float64:
    return np.sqrt(v.dot(v))


def normalize(v) -> NDArray[np.float64]:
    return v / magnitude(v)


def cross(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    return vector(a[1] * b[2] - a[2] * b[1],
                  a[2] * b[0] - a[0] * b[2],
                  a[0] * b[1] - a[1] * b[0])
