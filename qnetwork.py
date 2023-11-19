# Complete code for Flappy Bird game with Q-Learning and Experience Replay
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys

# Initialize Pygame and Q-Network
pygame.init()

# Hyperparameters
num_episodes = 50  # Increased number of episodes for training
buffer_size = 5000  # Increased Size of replay buffer
batch_size = 32  # Mini-batch size

# Initialize Replay Buffer
replay_buffer = []

# Q-Network definition
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize screen
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Initialize game variables
chr_x, chr_y = 350, 250
pipe_height = 300
pipe_x = 800
pipe_speed = 2
actions = [0, 1]  # 0: Do nothing, 1: Flap

# Initialize Q-Network and optimizer
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

q_network = QNetwork()
q_network.apply(init_weights)
optimizer = optim.Adam(q_network.parameters(), lr=0.0001)
criterion = nn.SmoothL1Loss()

# Initialize epsilon for epsilon-greedy action selection
epsilon = 0.3
epsilon_min = 0.01
epsilon_decay = 0.98  # Quicker decay

# Initialize reward
reward = 0

# Main training loop for multiple episodes
for episode in range(num_episodes):
    chr_x, chr_y = 350, 250
    pipe_height = 300
    pipe_x = 800
    reward = 0
    running = True
    pipe_height = random.randint(100, 400)

    while running:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit()

        state = torch.FloatTensor([chr_y / SCREEN_HEIGHT, (pipe_height - chr_y) / SCREEN_HEIGHT, (pipe_x - chr_x) / SCREEN_WIDTH])
        # state = torch.FloatTensor([chr_y, pipe_height - chr_y, pipe_x - chr_x])
        q_values = q_network(state)

        # Debugging: Print the Q-values and epsilon
        print(f"Q-values: {q_values}, Epsilon: {epsilon}")

        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = torch.argmax(q_values).item()

        # Debugging: Print the selected action
        print(f"Selected Action: {action}")

        if action == 1:
            chr_y = max(0, chr_y - 50)

        chr_y = min(SCREEN_HEIGHT - 50, chr_y + 1)

        pygame.draw.rect(screen, (0, 0, 255), (chr_x, chr_y, 50, 50))
        pygame.draw.rect(screen, (0, 255, 0), (pipe_x, 0, 100, pipe_height))
        pygame.draw.rect(screen, (0, 255, 0), (pipe_x, pipe_height + 200, 100, SCREEN_HEIGHT))

        pipe_x -= pipe_speed
        if pipe_x < -100:
            pipe_x = 800
            pipe_height = random.randint(100, 400)

        chr_rect = pygame.Rect(chr_x, chr_y, 50, 50)
        upper_pipe_rect = pygame.Rect(pipe_x, 0, 100, pipe_height)
        lower_pipe_rect = pygame.Rect(pipe_x, pipe_height + 200, 100, SCREEN_HEIGHT)

        # Update reward and Q-values
        if chr_rect.colliderect(upper_pipe_rect) or chr_rect.colliderect(lower_pipe_rect):
            reward = -1
            running = False
        elif pipe_x + 100 < chr_x:
            reward = 1
        elif chr_y <= 0 or chr_y >= SCREEN_HEIGHT - 50:  # Penalty for flying too high or too low
            reward = -0.5
        else:
            reward = 0

        new_state = torch.FloatTensor([chr_y, pipe_height - chr_y, pipe_x - chr_x])

        # Store experience in replay buffer
        experience = (state, action, reward, new_state)
        replay_buffer.append(experience)
        if len(replay_buffer) > buffer_size:
            replay_buffer.pop(0)

        # Sample a mini-batch of experiences from replay buffer
        if len(replay_buffer) >= batch_size:
            mini_batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, new_states = zip(*mini_batch)
            states_tensor = torch.stack(states)
            actions_tensor = torch.LongTensor(actions)
            rewards_tensor = torch.FloatTensor(rewards)
            new_states_tensor = torch.stack(new_states)

            q_values = q_network(states_tensor).gather(1, actions_tensor.view(-1, 1)).squeeze()
            next_q_values = q_network(new_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + 0.99 * next_q_values

            loss = criterion(q_values, target_q_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pygame.display.update()

    print(f"Episode {episode + 1} completed with epsilon {epsilon}.")
    # At the end of the episode, update epsilon
    epsilon = max(epsilon_min, epsilon_decay * epsilon)

pygame.quit()

# Example snippet for testing the trained Q-Learning agent in the Flappy Bird game.
# This assumes that you have a trained Q-Network model and the rest of your game code is in place.
print("TESTING!")
pygame.init()

# Initialize screen
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Example snippet for evaluating the trained Q-Learning agent in the Flappy Bird game.
# This assumes that you have a trained Q-Network model and the rest of your game code is in place.

# Number of test runs
num_tests = 5

# Initialize metrics
total_score = 0

# Set epsilon to 0 for full exploitation
epsilon = 0

# Main loop for multiple test runs
for test in range(num_tests):
    # Initialize game variables for each test run
    chr_x, chr_y = 350, 250
    pipe_height = 300
    pipe_x = 800
    score = 0  # Reset score for each test run
    win = 0  # Reset win flag for each test run

    # Main game loop for a single test run
    running = True
    while running:
        screen.fill((0, 0, 0))  # Fill screen with black
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Define state
        state = torch.FloatTensor([chr_y, pipe_height - chr_y, pipe_x - chr_x])

        # Get Q-values from Q-Network and choose action
        q_values = q_network(state)
        action = torch.argmax(q_values).item()

        # Execute action (Flap if action == 1)
        if action == 1:
            chr_y = max(0, chr_y - 50)

        # Update character position (Gravity)
        chr_y = min(SCREEN_HEIGHT - 50, chr_y + 1)

        # Collision detection
        chr_rect = pygame.Rect(chr_x, chr_y, 50, 50)
        upper_pipe_rect = pygame.Rect(pipe_x, 0, 100, pipe_height)
        lower_pipe_rect = pygame.Rect(pipe_x, pipe_height + 100, 100, SCREEN_HEIGHT)

        # Update reward and Q-values
        if chr_rect.colliderect(upper_pipe_rect) or chr_rect.colliderect(lower_pipe_rect):
            running = False  # End the game
        elif pipe_x + 100 < chr_x:
            score += 1  # Increment score


        # Draw character and pipes
        pygame.draw.rect(screen, (0, 0, 255), (chr_x, chr_y, 50, 50))
        pygame.draw.rect(screen, (0, 255, 0), (pipe_x, 0, 100, pipe_height))
        pygame.draw.rect(screen, (0, 255, 0), (pipe_x, pipe_height + 100, 100, SCREEN_HEIGHT))

        # Update pipe position
        pipe_x -= pipe_speed
        if pipe_x < -100:
            pipe_x = 800
            pipe_height = random.randint(100, 400)

        pygame.display.update()

    # Update metrics after each test run
    total_score += score
    print(f"Test {test + 1} completed with score {score}.")

pygame.quit()

# Note: This is a simplified example for testing the agent. 
# You would integrate this into your existing game code, replacing the training logic with this testing loop.

# Calculate average score and win rate
average_score = total_score / num_tests

print(f"Average Score: {average_score}")