
import pygame

from ghorman.color import color, pygame_color

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


def main():
    pygame.init()
    surface = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("ghorman demo")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                match event.dict['key']:
                    case pygame.K_RETURN:
                        running = False
                    case pygame.K_ESCAPE:
                        running = False

        surface.fill(pygame_color(color(0, 1, 1)))

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
