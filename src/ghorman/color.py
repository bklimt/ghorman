import numpy as np
import pygame

from numpy.typing import NDArray
from typing import overload


def color(r: float, g: float, b: float) -> NDArray[np.float64]:
    return np.array([r, g, b], dtype=np.float64)


def hadamard_product(c1: NDArray[np.float64], c2: NDArray[np.float64]) -> NDArray[np.float64]:
    return color(c1[0] * c2[0], c1[1] * c2[1], c1[2] * c2[2])


@overload
def pygame_color(c: NDArray[np.float64]) -> pygame.Color:
    ...


@overload
def pygame_color(c: pygame.Color) -> NDArray[np.float64]:
    ...


def pygame_color(c):
    if isinstance(c, pygame.Color):
        return color(c.r / 255, c.g / 255, c.b / 255)
    else:
        return pygame.Color((c[0] * 255, c[1] * 255, c[2] * 255))
