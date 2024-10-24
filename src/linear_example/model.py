"""
Multi-Layer Perceptron (MLP) implementation using PyTorch.

This module implements a flexible Multi-Layer Perceptron neural network
that can be configured with different numbers of layers and neurons.
The network uses ReLU activation functions between linear layers and
can be used for various supervised learning tasks.

References:
    - PyTorch Neural Networks: 
      https://pytorch.org/docs/stable/nn.html
    - PyTorch Linear Layer:
      https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    - PyTorch Sequential:
      https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
    - Neural Network Tutorial:
      https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

Example:
    >>> # Create a 3-layer MLP with 10 input features, 64 hidden units, and 2 outputs
    >>> model = MLP(input_size=10, hidden_size=64, output_size=2, num_layers=3)
    >>> batch_size, n_features = 32, 10
    >>> x = torch.randn(batch_size, n_features)
    >>> output = model(x)
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) implementation.
    
    This class implements a feedforward neural network with configurable layers
    and sizes. Each hidden layer is followed by a ReLU activation function.
    The network can be used for various tasks including classification and
    regression.

    Attributes:
        fcs (nn.Sequential): Sequential container of the network layers
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int
    ):
        """
        Initialize the MLP.

        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of neurons in each hidden layer
            output_size (int): Number of output features
            num_layers (int): Total number of layers (including input and output layers)

        Note:
            The network will have (num_layers - 2) hidden layers, with ReLU
            activation functions between them. The output layer does not have
            an activation function, allowing the network to be used for various
            tasks with different loss functions.
        """
        super().__init__()
        
        # Create list to hold network layers
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        
        # Add hidden layers
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
            
        # Add output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        # Combine all layers into a sequential model
        self.fcs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)

        Note:
            The input tensor must match the input_size specified during
            initialization. The output will have output_size features
            per sample in the batch.
        """
        return self.fcs(x)