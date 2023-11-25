from game.rendering import init_pygame, render_game_screen
from game.game_logic import init_game_variables, handle_events, update_game_state, check_collision_and_update_reward
from ai.q_network import init_q_network
from ai.training import update_q_values

import torch
import random
import pygame

def main():
    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
    screen = init_pygame()
    q_network, optimizer, criterion = init_q_network()
    num_episodes = 50  # Number of episodes for training
    replay_buffer = []  # Initialize the replay buffer
    buffer_size = 5000  # Size of the replay buffer
    batch_size = 32  # Mini-batch size
    gamma = 0.98  # Discount factor

    for episode in range(num_episodes):
        chr_x, chr_y, pipe_height, pipe_x, pipe_speed, actions, epsilon, epsilon_min, epsilon_decay, reward = init_game_variables()
        running = True

        while running:
            running = handle_events()
            state = torch.FloatTensor([chr_y / SCREEN_HEIGHT, (pipe_height - chr_y) / SCREEN_HEIGHT, (pipe_x - chr_x) / SCREEN_WIDTH])
            
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                q_values = q_network(state)
                action = torch.argmax(q_values).item()

            chr_y, pipe_x, pipe_height = update_game_state(action, chr_y, pipe_x, pipe_height)
            render_game_screen(screen, chr_x, chr_y, pipe_x, pipe_height)
            new_state, reward, done = check_collision_and_update_reward(chr_x, chr_y, pipe_x, pipe_height, SCREEN_WIDTH)

            experience = (torch.tensor(state, dtype=torch.float), action, reward, torch.tensor(new_state, dtype=torch.float))
            replay_buffer.append(experience)
            if len(replay_buffer) > buffer_size:
                replay_buffer.pop(0)

            update_q_values(replay_buffer, q_network, optimizer, criterion, batch_size, gamma)
            epsilon = max(epsilon_min, epsilon_decay * epsilon)

            if done:
                break

        print(f"Episode {episode + 1} completed. Epsilon: {epsilon}")

    pygame.quit()

if __name__ == "__main__":
    main()
