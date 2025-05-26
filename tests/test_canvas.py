
import pygame
import pytest

from typing import Generator

from ghorman.color import color, hadamard_product, pygame_color
from ghorman.utils import equal

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


@pytest.fixture
def pygame_surface() -> Generator[pygame.Surface]:
    pygame.init()
    surface = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    yield surface
    pygame.quit()


def test_pygame(pygame_surface: pygame.Surface):
    pygame_surface.fill(pygame_color(color(1, 0, 0)))
    assert pygame_surface.get_at((0, 0)) == pygame.Color((255, 0, 0))


def test_color_conversion():
    c1 = color(0.2, 0.4, 0.6)
    c2 = pygame.Color((51, 102, 153))
    assert pygame_color(c1) == c2
    assert equal(pygame_color(c2), c1)


def test_multiply_colors():
    c1 = color(1, 0.2, 0.4)
    c2 = color(0.9, 1, 0.1)
    assert equal(hadamard_product(c1, c2), color(0.9, 0.2, 0.04))
