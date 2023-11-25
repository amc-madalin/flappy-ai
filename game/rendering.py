import pygame
from utils.utils import load_config

# Load the configuration settings
config = load_config('configs/config.yml')

def init_pygame():
    pygame.init()
    screen_width = config['game_screen']['width']
    screen_height = config['game_screen']['height']
    screen = pygame.display.set_mode((screen_width, screen_height))
    return screen

def render_game_screen(screen, chr_x, chr_y, pipe_x, pipe_height):
    # Access configuration values
    pipe_width = config['game_mechanics']['pipe_width']
    pipe_gap = config['game_mechanics']['pipe_gap']
    character_size = config['game_mechanics']['character_size']
    screen_height = config['game_screen']['height']

    # Define colors
    black = (0, 0, 0)
    blue = (0, 0, 255)
    green = (0, 255, 0)

    # Clear screen
    screen.fill(black)

    # Draw character
    pygame.draw.rect(screen, blue, (chr_x, chr_y, character_size, character_size))

    # Draw pipes
    # Upper pipe
    pygame.draw.rect(screen, green, (pipe_x, 0, pipe_width, pipe_height))
    # Lower pipe
    pygame.draw.rect(screen, green, (pipe_x, pipe_height + pipe_gap, pipe_width, screen_height))

    # Update the display
    pygame.display.update()
