import argparse
import pygame
import torch.optim as optim
from model import QNetwork, init_weights
from train import game_loop
from test import test_model
import sys

def main():
    parser = argparse.ArgumentParser(description='Run Flappy Bird with Q-Learning.')
    parser.add_argument('--mode', type=str, help="Mode to run the program in ('train' or 'test')", required=True)
    
    args = parser.parse_args()
    
    # Initialize Pygame and screen
    pygame.init()
    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    
    # Initialize Q-Network and optimizer
    q_network = QNetwork()
    q_network.apply(init_weights)
    optimizer = optim.Adam(q_network.parameters(), lr=0.0001)
    
    # Initialize epsilon for epsilon-greedy action selection
    epsilon = 0.3
    
    if args.mode == 'train':
        game_loop(screen, q_network, optimizer, epsilon)
    elif args.mode == 'test':
        test_model(screen, q_network)
    else:
        print("Invalid mode selected.")
        sys.exit(1)

if __name__ == "__main__":
    main()
