import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys

# Initialize Pygame
def init_pygame(SCREEN_WIDTH, SCREEN_HEIGHT):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    return screen

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# Q-Network definition
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        # Define network layers
        self.fc1 = nn.Linear(3, 64)  # First fully connected layer
        self.fc2 = nn.Linear(64, 64)  # Second fully connected layer
        self.fc3 = nn.Linear(64, 32)  # Third fully connected layer
        self.fc4 = nn.Linear(32, 2)   # Output layer

    def forward(self, x):
        # Forward pass through the network
        x = torch.relu(self.fc1(x))  # Activation function after first layer
        x = torch.relu(self.fc2(x))  # Activation function after second layer
        x = torch.relu(self.fc3(x))  # Activation function after third layer
        x = self.fc4(x)              # No activation function after output layer
        return x


# Initialize Q-Network and optimizer
def init_q_network():
    q_network = QNetwork()
    q_network.apply(init_weights)
    optimizer = optim.Adam(q_network.parameters(), lr=0.0001)
    criterion = nn.SmoothL1Loss()
    return q_network, optimizer, criterion

# Initialize game variables
def init_game_variables():
    # Character initial position
    chr_x, chr_y = 350, 250

    # Initial pipe properties
    pipe_height = 300
    pipe_x = 800  # Initial horizontal position of the first pipe
    pipe_speed = 2  # Speed at which the pipes move leftward

    # Actions the character can take: [0, 1] (0: Do nothing, 1: Flap)
    actions = [0, 1]

    # Initialize epsilon for epsilon-greedy action selection
    epsilon = 0.3
    epsilon_min = 0.01
    epsilon_decay = 0.98  # Rate of decay for epsilon

    # Initialize reward
    reward = 0

    return chr_x, chr_y, pipe_height, pipe_x, pipe_speed, actions, epsilon, epsilon_min, epsilon_decay, reward


def handle_events():
    running = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                pygame.quit()
                sys.exit()
    return running


def update_game_state(action, chr_y, pipe_x, pipe_height):
    # Define constants
    GRAVITY = 1
    FLAP_STRENGTH = 50
    SCREEN_HEIGHT = 600
    PIPE_SPEED = 2

    # Update character's y-coordinate based on action
    if action == 1:  # Flap
        chr_y = max(0, chr_y - FLAP_STRENGTH)
    else:  # Gravity effect
        chr_y = min(SCREEN_HEIGHT - 50, chr_y + GRAVITY)

    # Update pipe's x-coordinate
    pipe_x -= PIPE_SPEED
    if pipe_x < -100:  # Reset pipe when it goes off screen
        pipe_x = 800
        pipe_height = random.randint(100, 400)  # Randomize new pipe height

    return chr_y, pipe_x, pipe_height


def render_game_screen(screen, chr_x, chr_y, pipe_x, pipe_height):
    # Define colors
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    PIPE_WIDTH = 100
    PIPE_GAP = 200
    CHARACTER_SIZE = 50
    SCREEN_HEIGHT = 600

    # Clear screen
    screen.fill(BLACK)

    # Draw character
    pygame.draw.rect(screen, BLUE, (chr_x, chr_y, CHARACTER_SIZE, CHARACTER_SIZE))

    # Draw pipes
    # Upper pipe
    pygame.draw.rect(screen, GREEN, (pipe_x, 0, PIPE_WIDTH, pipe_height))
    # Lower pipe
    pygame.draw.rect(screen, GREEN, (pipe_x, pipe_height + PIPE_GAP, PIPE_WIDTH, SCREEN_HEIGHT))

    # Update the display
    pygame.display.update()


def update_q_values(replay_buffer, q_network, optimizer, criterion, batch_size, gamma):
    # Check if the replay buffer has enough samples
    if len(replay_buffer) < batch_size:
        return

    # Sample a mini-batch of experiences from the replay buffer
    mini_batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states = zip(*mini_batch)

    # Convert to tensors
    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float)
    next_states = torch.stack(next_states)

    # Get current Q values from the network
    current_q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute the expected Q values
    max_next_q_values = q_network(next_states).detach().max(1)[0]
    expected_q_values = rewards + (gamma * max_next_q_values)

    # Compute loss
    loss = criterion(current_q_values, expected_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def check_collision_and_update_reward(chr_x, chr_y, pipe_x, pipe_height, SCREEN_WIDTH):
    # Constants
    CHARACTER_SIZE = 50
    PIPE_WIDTH = 100
    SCREEN_HEIGHT = 600
    COLLISION_PENALTY = -10  # Negative reward for collision
    PASS_REWARD = 1  # Positive reward for passing a pipe
    HEIGHT_PENALTY = -0.1  # Small penalty for being too high or too low

    # Check for collision with upper pipe
    if chr_x < pipe_x + PIPE_WIDTH and chr_x + CHARACTER_SIZE > pipe_x and chr_y < pipe_height:
        reward = COLLISION_PENALTY
        done = True
    # Check for collision with lower pipe
    elif chr_x < pipe_x + PIPE_WIDTH and chr_x + CHARACTER_SIZE > pipe_x and chr_y + CHARACTER_SIZE > pipe_height + 200:
        reward = COLLISION_PENALTY
        done = True
    # Check if the character has successfully passed the pipe
    elif pipe_x + PIPE_WIDTH < chr_x:
        reward = PASS_REWARD
        done = False
    # Penalty for flying too high or too low
    elif chr_y <= 0 or chr_y >= SCREEN_HEIGHT - CHARACTER_SIZE:
        reward = HEIGHT_PENALTY
        done = True
    else:
        reward = 0
        done = False

    new_state = [chr_y / SCREEN_HEIGHT, (pipe_height - chr_y) / SCREEN_HEIGHT, (pipe_x - chr_x) / SCREEN_WIDTH]

    return new_state, reward, done


def main():
    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
    screen = init_pygame(SCREEN_WIDTH, SCREEN_HEIGHT)
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

