# Project Structure

This document outlines the folder and file structure of the project.

## Main Application

- `main.py`: The entry point of the application where the main function is located.

## Game Module

This folder contains modules related to the game logic and rendering.

- `game/`
  - `config.py`: Contains game configurations like screen dimensions, gravity, etc.
  - `game_logic.py`: Handles the game state updates, including character movements and pipe positions.
  - `rendering.py`: Manages the rendering of the game screen, including drawing characters and pipes.

## AI Module

Dedicated to AI and neural network functionalities.

- `ai/`
  - `q_network.py`: Contains the QNetwork class definition and its methods.
  - `training.py`: Manages the training loop, including updating the Q-values.
  - `replay_buffer.py`: Manages the replay buffer's storage and retrieval of experiences.

## Utilities

Contains utility functions or classes that can be used across the project.

- `utils/`
  - `utils.py`: General utility functions like weight initialization.

## Tests

Includes unit tests for your application.

- `tests/`
  - `test_game.py`: Test cases for the game logic to ensure it works as expected.

## Additional Files

- `environment.yml`: Lists all the Python dependencies required for your project.
- `README.md`: A Markdown file providing an overview of your project, how to set it up, and how to run it.
