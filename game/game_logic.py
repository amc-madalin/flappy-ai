import sys
import torch
import random
import pygame

def init_game_variables(config):
    # Using values from configuration
    chr_x = config['initial_positions']['character_x']
    chr_y = config['initial_positions']['character_y']
    
    # Randomize the starting pipe height within the specified range
    pipe_height_min = config['initial_positions']['pipe_height_min']
    pipe_height_max = config['initial_positions']['pipe_height_max']
    
    pipe_height = random.randint(pipe_height_min, pipe_height_max)
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

def update_game_state(action, chr_y, pipe_x, pipe_height, config):
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

import torch

def get_state_representation(chr_x, chr_y, pipe_x, pipe_height, config):
    screen_height = config['game_screen']['height']
    screen_width = config['game_screen']['width']
    pipe_gap = config['game_mechanics']['pipe_gap']
    character_size = config['game_mechanics']['character_size']

    # Normalized vertical position of the character
    normalized_chr_y = chr_y / screen_height

    # Distance from the character's top to the bottom edge of the upper pipe
    distance_to_upper_pipe = pipe_height - (chr_y + character_size)

    # Distance from the character's bottom to the top edge of the lower pipe
    lower_pipe_top_edge = pipe_height + pipe_gap
    distance_to_lower_pipe = chr_y - lower_pipe_top_edge

    # Horizontal Distance to the Next Pipe
    distance_to_next_pipe = pipe_x - chr_x

    # Normalizing the distances
    normalized_distance_to_upper_pipe = distance_to_upper_pipe / screen_height
    normalized_distance_to_lower_pipe = distance_to_lower_pipe / screen_height
    normalized_distance_to_next_pipe = distance_to_next_pipe / screen_width

    # Create state tensor
    state = torch.FloatTensor([
        normalized_chr_y, 
        normalized_distance_to_upper_pipe, 
        normalized_distance_to_lower_pipe, 
        normalized_distance_to_next_pipe
    ])

    return state


def check_collision_and_update_reward(chr_x, chr_y, pipe_x, pipe_height, config):
    # Using values from configuration
    character_size = config['game_mechanics']['character_size']
    pipe_width = config['game_mechanics']['pipe_width']
    pipe_gap = config['game_mechanics']['pipe_gap']
    screen_height = config['game_screen']['height']
    screen_width = config['game_screen']['width']
    collision_penalty = config['game_mechanics']['collision_penalty']
    pass_reward = config['game_mechanics']['pass_reward']
    height_penalty = config['game_mechanics']['height_penalty']

    # Check for collision with upper pipe
    if chr_x < pipe_x + pipe_width and chr_x + character_size > pipe_x and chr_y < pipe_height:
        reward = collision_penalty
        done = True
    # Check for collision with lower pipe
    elif chr_x < pipe_x + pipe_width and chr_x + character_size > pipe_x and chr_y + character_size > pipe_height + pipe_gap:
        reward = collision_penalty
        done = True
    # Check if the character has successfully passed the pipe
    elif pipe_x + pipe_width < chr_x:
        reward = pass_reward
        done = False
    # Penalty for flying too high or too low
    elif chr_y <= 0 or chr_y >= screen_height - character_size:
        reward = height_penalty
        done = False
    else:
        reward = 0
        done = False

    # Use the existing state representation function
    new_state = get_state_representation(chr_x, chr_y, pipe_x, pipe_height, config)

    return new_state, reward, done
