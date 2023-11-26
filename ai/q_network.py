import torch
import torch.nn as nn
import torch.optim as optim
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        # Define network layers
        self.fc1 = nn.Linear(4, 64)  # First fully connected layer
        self.fc2 = nn.Linear(64, 64)  # Second fully connected layer
        self.fc3 = nn.Linear(64, 32)  # Third fully connected layer
        self.fc4 = nn.Linear(32, 2)   # Output layer

        # Define LeakyReLU activation
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        # Forward pass through the network using LeakyReLU
        x = self.leaky_relu(self.fc1(x))  # Activation function after first layer
        x = self.leaky_relu(self.fc2(x))  # Activation function after second layer
        x = self.leaky_relu(self.fc3(x))  # Activation function after third layer
        x = self.fc4(x)                   # No activation function after output layer
        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# Initialize Q-Network and optimizer
def init_q_network():
    q_network = QNetwork()
    q_network.apply(init_weights)
    optimizer = optim.Adam(q_network.parameters(), lr=0.0001)
    criterion = nn.SmoothL1Loss()
    return q_network, optimizer, criterion