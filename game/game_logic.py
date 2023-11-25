import sys
import random
import pygame
from utils.utils import load_config

# Load configuration
config = load_config('configs/config.yml')

def init_game_variables():
    # Using values from configuration
    chr_x = config['initial_positions']['character_x']
    chr_y = config['initial_positions']['character_y']
    pipe_height = config['initial_positions']['pipe_height']
    pipe_x = config['initial_positions']['pipe_x']
    pipe_speed = config['game_mechanics']['pipe_speed']
    actions = [0, 1]  # Actions remain the same

    # AI-related configuration
    epsilon = config['ai_config']['epsilon']
    epsilon_min = config['ai_config']['epsilon_min']
    epsilon_decay = config['ai_config']['epsilon_decay']
    reward = 0  # Initialize reward

    return chr_x, chr_y, pipe_height, pipe_x, pipe_speed, actions, epsilon, epsilon_min, epsilon_decay, reward

def handle_events():
    # Event handling remains the same
    running = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                pygame.quit()
                sys.exit()
    return running

def update_game_state(action, chr_y, pipe_x, pipe_height):
    # Using values from configuration
    gravity = config['game_mechanics']['gravity']
    flap_strength = config['game_mechanics']['flap_strength']
    screen_height = config['game_screen']['height']
    pipe_speed = config['game_mechanics']['pipe_speed']

    # Update character's y-coordinate based on action
    if action == 1:  # Flap
        chr_y = max(0, chr_y - flap_strength)
    else:  # Gravity effect
        chr_y = min(screen_height - 50, chr_y + gravity)

    # Update pipe's x-coordinate
    pipe_x -= pipe_speed
    if pipe_x < -100:  # Reset pipe when it goes off screen
        pipe_x = config['initial_positions']['pipe_x']
        pipe_height = random.randint(100, 400)  # Randomize new pipe height

    return chr_y, pipe_x, pipe_height

def check_collision_and_update_reward(chr_x, chr_y, pipe_x, pipe_height, screen_width):
    # Using values from configuration
    character_size = config['game_mechanics']['character_size']
    pipe_width = config['game_mechanics']['pipe_width']
    screen_height = config['game_screen']['height']
    collision_penalty = -10  # These could also be moved to config
    pass_reward = 1
    height_penalty = -0.1

    # Check for collision with upper pipe
    if chr_x < pipe_x + pipe_width and chr_x + character_size > pipe_x and chr_y < pipe_height:
        reward = collision_penalty
        done = True
    # Check for collision with lower pipe
    elif chr_x < pipe_x + pipe_width and chr_x + character_size > pipe_x and chr_y + character_size > pipe_height + 200:
        reward = collision_penalty
        done = True
    # Check if the character has successfully passed the pipe
    elif pipe_x + pipe_width < chr_x:
        reward = pass_reward
        done = False
    # Penalty for flying too high or too low
    elif chr_y <= 0 or chr_y >= screen_height - character_size:
        reward = height_penalty
        done = True
    else:
        reward = 0
        done = False

    new_state = [chr_y / screen_height, (pipe_height - chr_y) / screen_height, (pipe_x - chr_x) / screen_width]

    return new_state, reward, done
