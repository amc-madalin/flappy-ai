import sys
import torch
import pygame
import random

def initialize_game_variables():
    chr_x, chr_y = 350, 250
    pipe_height = random.randint(100, 400)
    pipe_x = 800
    pipe_speed = 2
    return chr_x, chr_y, pipe_height, pipe_x, pipe_speed

def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit()
    return True

def select_action(state, model, epsilon):
    q_values = model(state)
    if random.random() < epsilon:
        return random.choice([0, 1])
    else:
        return torch.argmax(q_values).item()

def update_state(chr_y, pipe_height, pipe_x, chr_x, SCREEN_HEIGHT, SCREEN_WIDTH, action):
    if action == 1:
        chr_y = max(0, chr_y - 50)
    chr_y = min(SCREEN_HEIGHT - 50, chr_y + 1)
    new_state = torch.FloatTensor([chr_y / SCREEN_HEIGHT, (pipe_height - chr_y) / SCREEN_HEIGHT, (pipe_x - chr_x) / SCREEN_WIDTH])
    return new_state

def update_pipe_position(pipe_x, pipe_speed):
    pipe_x -= pipe_speed
    if pipe_x < -100:
        pipe_x = 800
        pipe_height = random.randint(100, 400)
    return pipe_x, pipe_height