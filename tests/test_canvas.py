
import os
import pygame
import pytest

from typing import Generator

from ghorman.canvas import pixel_at, write_pixel
from ghorman.color import color, hadamard_product, pygame_color
from ghorman.utils import equal, normalize, point, vector

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


@pytest.fixture
def pygame_init():
    pygame.init()
    yield None
    pygame.quit()


def test_canvas(pygame_init: None):
    canvas = pygame.Surface((100, 100))

    red = pygame.Color(255, 0, 0)
    canvas.fill(red)
    assert pixel_at(canvas, 0, 0) == red

    teal = pygame.Color(0, 127, 127)
    write_pixel(canvas, 20, 30, teal)
    assert pixel_at(canvas, 20, 30) == teal


def test_color_conversion():
    c1 = color(0.2, 0.4, 0.6)
    c2 = pygame.Color((51, 102, 153))
    assert pygame_color(c1) == c2
    assert equal(pygame_color(c2), c1)


def test_multiply_colors():
    c1 = color(1, 0.2, 0.4)
    c2 = color(0.9, 1, 0.1)
    assert equal(hadamard_product(c1, c2), color(0.9, 0.2, 0.04))


def test_save_canvas(pygame_init: None):
    width = 900
    height = 550

    start = point(0, 1, 0)
    velocity = normalize(vector(1, 1.8, 0)) * 11.25
    gravity = vector(0, -0.1, 0)
    wind = vector(-0.01, 0, 0)

    color = pygame.Color(255, 0, 0)
    background = pygame.Color(0, 0, 0)

    canvas = pygame.Surface((width, height))
    canvas.fill(background)
    p = start
    while p[0] < width:
        x = int(p[0])
        y = (height - 1) - int(p[1])
        if x >= 0 and x < width and y >= 0 and y < height:
            write_pixel(canvas, x, y, color)

        p += velocity
        velocity += wind
        velocity += gravity

    os.mkdir('./tmp')
    pygame.image.save(canvas, './tmp/save_canvas.png')
