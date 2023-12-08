# main.py
from game.rendering import init_pygame, render_game_screen
from game.game_logic import init_game_variables, handle_events, update_game_state, check_collision_and_update_reward, get_state_representation
from ai.q_network import init_q_network
from ai.training import update_q_values
from utils.utils import load_config

import torch
import random
import pygame

def main():
    # Load configuration
    config = load_config('configs/config.yml')

    # Use values from configuration
    screen = init_pygame(config['game_screen']['width'], config['game_screen']['height'])
    q_network, optimizer, criterion = init_q_network()  # Assuming this doesn't need config
    num_episodes = config['ai_config']['num_episodes']
    buffer_size = config['ai_config']['buffer_size']
    save_path = config['model']['save_path']
    replay_buffer = []

    for episode in range(num_episodes):
        chr_x, chr_y, pipe_height, pipe_x, pipe_speed, actions, epsilon, epsilon_min, epsilon_decay, reward = init_game_variables(config)
        running = True

        while running:
            running = handle_events()
            state = get_state_representation(
                chr_x, chr_y, pipe_x, pipe_height, config
                )
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                q_values = q_network(state)
                action = torch.argmax(q_values).item()

            chr_y, pipe_x, pipe_height = update_game_state(action, chr_y, pipe_x, pipe_height, config)
            render_game_screen(screen, chr_x, chr_y, pipe_x, pipe_height, config)
            new_state, reward, done = check_collision_and_update_reward(chr_x, chr_y, pipe_x, pipe_height, config)

            experience = (torch.tensor(state, dtype=torch.float), action, reward, torch.tensor(new_state, dtype=torch.float))
            replay_buffer.append(experience)
            if len(replay_buffer) > buffer_size:
                replay_buffer.pop(0)

            update_q_values(replay_buffer, q_network, optimizer, criterion, config)
            epsilon = max(epsilon_min, epsilon_decay * epsilon)
            
            

            if done:
                break

        print(f"Episode {episode + 1} completed. Epsilon: {epsilon}")

    # Save the model and training state
    torch.save({
        'episode': episode + 1,
        'model_state_dict': q_network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'replay_buffer': replay_buffer
    }, save_path)
    
    pygame.quit()

if __name__ == "__main__":
    main()
