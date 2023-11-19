import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys

def initialize_game():
    pygame.init()
    return pygame.display.set_mode((800, 600))

def initialize_q_network():
    model = QNetwork()
    model.apply(init_weights)
    return model

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

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

def game_loop(screen, model, optimizer, epsilon):
    """
    Main game loop function.

    Args:
        screen: Pygame screen object.
        model: PyTorch model object.
        optimizer: PyTorch optimizer object.
        epsilon: Epsilon value for epsilon-greedy policy.

    Returns:
        replay_buffer: List containing replay buffer data.
    """
    # Initialize game variables
    chr_x, chr_y = 350, 250
    pipe_height = 300
    pipe_x = 800
    pipe_speed = 2
    actions = [0, 1]  # 0: Do nothing, 1: Flap
    reward = 0
    running = True
    replay_buffer = []
    batch_size = 32
    buffer_size = 5000
    criterion = nn.SmoothL1Loss()
    
    while running:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        
        state = torch.FloatTensor([chr_y, pipe_height - chr_y, pipe_x - chr_x])
        q_values = model(state)

        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = torch.argmax(q_values).item()

        if action == 1:
            chr_y = max(0, chr_y - 50)
        chr_y = min(SCREEN_HEIGHT - 50, chr_y + 1)
        
        # Update and draw game objects
        pygame.draw.rect(screen, (0, 0, 255), (chr_x, chr_y, 50, 50))
        pygame.draw.rect(screen, (0, 255, 0), (pipe_x, 0, 100, pipe_height))
        pygame.draw.rect(screen, (0, 255, 0), (pipe_x, pipe_height + 200, 100, SCREEN_HEIGHT))
        
        pipe_x -= pipe_speed
        if pipe_x < -100:
            pipe_x = 800
            pipe_height = random.randint(100, 400)
        
        new_state = torch.FloatTensor([chr_y, pipe_height - chr_y, pipe_x - chr_x])
        experience = (state, action, reward, new_state)
        replay_buffer.append(experience)
        
        if len(replay_buffer) > buffer_size:
            replay_buffer.pop(0)
        
        if len(replay_buffer) >= batch_size:
            # Perform one step of optimization
            mini_batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, new_states = zip(*mini_batch)
            states_tensor = torch.stack(states)
            actions_tensor = torch.LongTensor(actions)
            rewards_tensor = torch.FloatTensor(rewards)
            new_states_tensor = torch.stack(new_states)

            q_values = model(states_tensor).gather(1, actions_tensor.view(-1, 1)).squeeze()
            next_q_values = model(new_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + 0.99 * next_q_values

            loss = criterion(q_values, target_q_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        pygame.display.update()
    return replay_buffer  # Return updated replay buffer for further use


def test_model(screen, model):
    """
    Function for testing the trained model.

    Args:
        screen: Pygame screen object.
        model: PyTorch model object.

    Returns:
        total_score: The total score achieved in the test run.
    """
    # Initialize game variables
    chr_x, chr_y = 350, 250
    pipe_height = 300
    pipe_x = 800
    pipe_speed = 2
    running = True
    score = 0  # Reset score for the test run
    
    while running:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        state = torch.FloatTensor([chr_y, pipe_height - chr_y, pipe_x - chr_x])
        q_values = model(state)
        action = torch.argmax(q_values).item()
        
        if action == 1:
            chr_y = max(0, chr_y - 50)
        chr_y = min(SCREEN_HEIGHT - 50, chr_y + 1)
        
        # Update and draw game objects
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
        
        if chr_rect.colliderect(upper_pipe_rect) or chr_rect.colliderect(lower_pipe_rect):
            running = False
        elif pipe_x + 100 < chr_x:
            score += 1  # Increment score

        pygame.display.update()
        
    return score  # Return the total score for the test run


def main():
    screen = initialize_game()
    q_network = initialize_q_network()
    optimizer = optim.Adam(q_network.parameters(), lr=0.0001)
    criterion = nn.SmoothL1Loss()
    epsilon = 0.3  # Initial epsilon

    for episode in range(50):
        game_loop(screen, q_network, optimizer, epsilon)
        epsilon *= 0.98  # Epsilon decay

    test_model(screen, q_network)

if __name__ == "__main__":
    main()
