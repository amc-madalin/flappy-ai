import torch
from game.rendering import init_pygame, render_game_screen
from game.game_logic import handle_events, update_game_state, check_collision_and_update_reward, get_state_representation, init_game_variables
from ai.q_network import QNetwork
from utils.utils import load_config
import pygame

def test_game(model_path, config):
    # Initialize Pygame and the screen
    screen = init_pygame(config['game_screen']['width'], config['game_screen']['height'])

    # Initialize the Q-network
    q_network = QNetwork()
    
    # Load the saved model state
    checkpoint = torch.load(model_path)
    q_network.load_state_dict(checkpoint['model_state_dict'])

    # Ensure the network is in evaluation mode
    q_network.eval()

    # Initialize game variables
    chr_x, chr_y, pipe_height, pipe_x, pipe_speed, _, _, _, _, _ = init_game_variables(config)
    
    running = True
    while running:
        # Handle events like game exit
        running = handle_events()

        # Get the current state representation
        state = get_state_representation(chr_x, chr_y, pipe_x, pipe_height, config)

        # Forward pass through the network to get Q-values for the current state
        with torch.no_grad():  # We do not need gradient computation for testing
            q_values = q_network(state)
        
        # Select the action with the highest Q-value
        action = torch.argmax(q_values).item()

        # Update the game state based on the selected action
        chr_y, pipe_x, pipe_height = update_game_state(action, chr_y, pipe_x, pipe_height, config)

        # Render the game screen
        render_game_screen(screen, chr_x, chr_y, pipe_x, pipe_height, config)

        # Check for collision and update reward (for testing, we may not need the reward)
        _, _, done = check_collision_and_update_reward(chr_x, chr_y, pipe_x, pipe_height, config)

        if done:
            # If the game is over, exit the loop
            break

    pygame.quit()

if __name__ == "__main__":
    # Load configuration
    config = load_config('configs/config.yml')

    # Specify the path to your saved model
    model_path = './model/best_bird.pth'

    # Call the test game function
    test_game(model_path, config)
