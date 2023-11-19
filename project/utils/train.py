import torch
import random

def store_experience(replay_buffer, state, action, reward, new_state, buffer_size):
    experience = (state, action, reward, new_state)
    replay_buffer.append(experience)
    if len(replay_buffer) > buffer_size:
        replay_buffer.pop(0)

def train_q_network(replay_buffer, batch_size, model, optimizer, criterion):
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

def compute_reward(chr_rect, upper_pipe_rect, lower_pipe_rect, pipe_x, chr_x, chr_y, SCREEN_HEIGHT):
    if chr_rect.colliderect(upper_pipe_rect) or chr_rect.colliderect(lower_pipe_rect):
        return -1, False
    elif pipe_x + 100 < chr_x:
        return 1, True
    elif chr_y <= 0 or chr_y >= SCREEN_HEIGHT - 50:
        return -0.5, True
    else:
        return 0, True