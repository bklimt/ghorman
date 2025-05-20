
from math import isclose
from numpy import array, float64
from numpy.typing import NDArray
from typing import Tuple

EPSILON = 0.00001


def point(x: float, y: float, z: float) -> NDArray[float64]:
    return array([x, y, z, 1.0])


def vector(x: float, y: float, z: float) -> NDArray[float64]:
    return array([x, y, z, 0.0])


def equal(f1: float, f2: float) -> bool:
    return isclose(f1, f2, abs_tol=EPSILON)
