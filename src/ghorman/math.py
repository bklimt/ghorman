
import numpy as np

from numpy.typing import NDArray
from typing import List, overload

EPSILON = 0.00001


def point(x: float, y: float, z: float) -> NDArray[np.float64]:
    return np.array([x, y, z, 1.0], dtype=np.float64)


def vector(x: float, y: float, z: float) -> NDArray[np.float64]:
    return np.array([x, y, z, 0.0], dtype=np.float64)


def matrix(m: List[List[float]]) -> NDArray[np.float64]:
    return np.array(m, dtype=np.float64, ndmin=2)


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


def multiply(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.matmul(a, b)


def identity() -> NDArray[np.float64]:
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)


def det(m: NDArray[np.float64]) -> np.float64:
    return np.linalg.det(m)


def submatrix(m: NDArray[np.float64], rowToRemove: int, columnToRemove: int) -> NDArray[np.float64]:
    rows: List[List[np.float64]] = []
    for i, row in enumerate(m):
        if i != rowToRemove:
            rows.append([element for j, element in enumerate(
                row) if j != columnToRemove])
    return np.array(rows, dtype=np.float64)


def minor(m: NDArray[np.float64], row: int, column: int) -> np.float64:
    return det(submatrix(m, row, column))
