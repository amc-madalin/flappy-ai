import random
import torch

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