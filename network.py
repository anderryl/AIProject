import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Net(nn.Module):
    """
    Neural network class for a classifier (with a default of two labels)
    """
    def __init__(self, inputs: int, layers: [int], activation: nn.Module | None, outputs: int = 2):
        """
        Initializes a classifier NN from given inputs and layers and activation function
        :param inputs: number of input features
        :param layers: list of nueron counts for each hidden layer
        :param activation: activation function (defaults to None for linear activation function)
        :param outputs: number of class labels (defaults to 2)
        """
        super(Net, self).__init__()
        self.flatten = nn.Flatten()

        # Build ANN module sequence as alternating between Linear connections (weights/biases) and activation functions
        stack = OrderedDict[str, nn.Module]()
        full = layers + [outputs]
        stack['in'] = nn.Linear(inputs, full[0])
        last = full[0]
        if len(full) > 1:
            for layer in full[1:]:
                if activation is not None:
                    stack["activation_" + str(layer)] = activation
                stack["connection_" + str(layer)] = nn.Linear(last, layer)
                last = layer
        self.stack = nn.Sequential(stack)

        # Determine what hardware is available
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.to(device)

    def forward(self, input):
        """
        Finds unnormalized model output for a given input
        :param input: input vector
        :return: output vector
        """
        x = self.flatten(input)
        logits = self.stack(x)
        return logits


