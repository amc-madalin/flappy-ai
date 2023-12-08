# Flappy-AI with Reinforcement Learning

This is a project that uses AI to train an agent to play a game. The project is structured as follows:

- `ai/`: Contains the code for the AI agent, including the Q network, replay buffer, and training logic.
- `assets/`: Contains any assets used by the project.
- `configs/`: Contains configuration files.
- `environment.yml`: Contains the list of dependencies for the project.
- `experiments/`: Contains any experimental code or data.
- `game/`: Contains the game logic and rendering code.
- `model/`: Contains the trained model.
- `runs/`: Contains the output from training runs.
- `utils/`: Contains utility code used across the project.
- `train.py`: The main script to start training the agent.
- `test_game.py`: The script to test the game without the agent.

## Installation

1. Clone the repository:

```sh
git clone https://github.com/amc-madalin/flappy-ai.git
```

2. Navigate to the project directory:

```sh
cd flappy-ai
```

3. Install the dependencies:

```sh
conda env create -f environment.yml
```

## Usage

Activate environment:

```sh
conda activate flappy-ai
```

To train the agent, run:

```sh
python train.py
```

To test the game, run:

```sh
python test_game.py
```

## License

This project is licensed under the MIT License.
