import yaml
import torch
import torch.nn as nn
from ai.q_network import QNetwork  # Import the QNetwork class

def load_model(filepath):
    """
    Load a trained Q-Network model from the specified file.

    Args:
    filepath (str): Path to the file containing the saved model.

    Returns:
    QNetwork: The loaded Q-Network model.
    """
    # Ensure that a Q-Network model can be created here
    model = QNetwork()
    
    # Load the saved state_dict into the model
    model.load_state_dict(torch.load(filepath))
    
    # Set the model to evaluation mode
    model.eval()
    
    return model


def load_config(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config
