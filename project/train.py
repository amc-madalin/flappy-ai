import torch
import sys
import pygame
import random
import json

import utils.game
import utils.train

# Load hyperparameters from JSON file
with open("hyperparams.json", "r") as f:
    config = json.load(f)

# Assuming you've already initialized screen, q_network, optimizer, criterion, replay_buffer
config['screen'] = screen
config['q_network'] = q_network
config['optimizer'] = optimizer
config['criterion'] = criterion
config['replay_buffer'] = replay_buffer

def main_training_loop(config):
    for episode in range(config['num_episodes']):
        chr_x, chr_y, pipe_height, pipe_x, pipe_speed = utils.game.initialize_game_variables()
        running = True

        while running:
            running = utils.game.handle_events()
            config['screen'].fill((0, 0, 0))
            
            state = torch.FloatTensor([chr_y / config['SCREEN_HEIGHT'], (pipe_height - chr_y) / config['SCREEN_HEIGHT'], (pipe_x - chr_x) / config['SCREEN_WIDTH']])
            action = utils.game.select_action(state, config['q_network'], config['epsilon'])
            
            new_state = utils.game.update_state(chr_y, pipe_height, pipe_x, chr_x, config['SCREEN_HEIGHT'], action)
            draw_game_screen(config['screen'], chr_x, chr_y, pipe_x, pipe_height, config['SCREEN_HEIGHT'])
            
            chr_rect = pygame.Rect(chr_x, chr_y, 50, 50)
            upper_pipe_rect = pygame.Rect(pipe_x, 0, 100, pipe_height)
            lower_pipe_rect = pygame.Rect(pipe_x, pipe_height + 200, 100, config['SCREEN_HEIGHT'])
            
            reward, running = utils.train.compute_reward(chr_rect, upper_pipe_rect, lower_pipe_rect, pipe_x, chr_x, chr_y, config['SCREEN_HEIGHT'])
            
            utils.train.store_experience(config['replay_buffer'], state, action, reward, new_state, config['buffer_size'])
            
            if len(config['replay_buffer']) >= config['batch_size']:
                utils.train.train_q_network(config['replay_buffer'], config['batch_size'], config['q_network'], config['optimizer'], config['criterion'])
            
            pipe_x, pipe_height = utils.game.update_pipe_position(pipe_x, pipe_speed)
            config['epsilon'] = max(config['epsilon_min'], config['epsilon_decay'] * config['epsilon'])
