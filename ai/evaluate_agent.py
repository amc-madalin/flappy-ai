# File: ai/evaluate_agent.py
import pygame
import torch
from game.rendering import init_pygame, render_game_screen
from game.game_logic import init_game_variables, update_game_state, check_collision_and_update_reward, handle_events
from utils.utils import load_model  # Assuming you have a utility function to load the trained model

print("TESTING!")
screen = init_pygame(SCREEN_WIDTH, SCREEN_HEIGHT)

# Load the trained Q-Network model
q_network = load_model("path_to_trained_model")

# Number of test runs and initial metrics
num_tests = 5
total_score = 0

# Main loop for multiple test runs
for test in range(num_tests):
    # Initialize game variables for each test run
    chr_x, chr_y, pipe_height, pipe_x, pipe_speed, actions, epsilon, epsilon_min, epsilon_decay, reward = init_game_variables()
    score = 0  # Reset score for each test run

    # Main game loop for a single test run
    running = True
    while running:
        running = handle_events()
        state = torch.FloatTensor([chr_y / SCREEN_HEIGHT, (pipe_height - chr_y) / SCREEN_HEIGHT, (pipe_x - chr_x) / SCREEN_WIDTH])

        # Get Q-values from Q-Network and choose action
        q_values = q_network(state)
        action = torch.argmax(q_values).item()

        # Update game state and render
        chr_y, pipe_x, pipe_height = update_game_state(action, chr_y, pipe_x, pipe_height)
        render_game_screen(screen, chr_x, chr_y, pipe_x, pipe_height)

        # Check collision and update reward
        new_state, reward, done = check_collision_and_update_reward(chr_x, chr_y, pipe_x, pipe_height, SCREEN_WIDTH)
        if done:
            running = False
        elif pipe_x + 100 < chr_x:
            score += 1  # Increment score

    # Update metrics after each test run
    total_score += score
    print(f"Test {test + 1} completed with score {score}.")

pygame.quit()

# Calculate average score
average_score = total_score / num_tests
print(f"Average Score: {average_score}")
