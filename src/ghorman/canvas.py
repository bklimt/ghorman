
import numpy as np
import pygame

from ghorman.color import color, pygame_color
from numpy.typing import NDArray


def write_pixel(canvas: pygame.Surface, x: int, y: int, color: pygame.Color | NDArray[np.float64]):
    if not isinstance(color, pygame.Color):
        color = pygame_color(color)
    canvas.set_at((x, y), color)


def pixel_at(canvas: pygame.Surface, x: int, y: int) -> pygame.Color:
    return canvas.get_at((x, y))
