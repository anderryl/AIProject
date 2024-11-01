# Documentation statement: heavily used and modified code from pytorch's neural network documentation section

from expression import Exp
from loading import stats, create_datasets
from network import Net
import numpy as np
from learning import train
import torch.nn as nn

# Statistic expressions to be calculated
expressions = [
    Exp("complete_pass") / Exp("pass") // "completion_rate",
    Exp("rushing_yards") / Exp("rush") // "rushing_avg",
    Exp("receiving_yards") / Exp("pass") // "passing_avg",
    Exp("penalty") / Exp("plays") // "penalty_avg",
    Exp("sack") / Exp("plays") // "sack_rate",
    Exp("fumble") / Exp("plays") // "fumble_rate",
    Exp("interception") / Exp("pass") // "intercept_rate",
    Exp("yards_gained") / Exp("drives") // "drive_yards",
    Exp("first_down") / Exp("drives") // "drive_firsts",
]

input_length = 4 * len(expressions)

# Print final results
data = stats(expressions)
print("Loading complete")
relu = []
sigmoid = []
linear = []

# Resample training, validation, and testing data 100 times
for i in range(100):
    # Resample the data
    training, validation, testing = create_datasets(data)
    # ReLU
    network = Net(input_length, [input_length], nn.ReLU())
    relu.append(train(network, training, validation, testing, 0.9e-3, 10, 15, 1e-3))
    # Sigmoid
    network = Net(input_length, [input_length], nn.Sigmoid())
    sigmoid.append(train(network, training, validation, testing, 0.9e-3, 10, 15, 0))
    # Linear
    network = Net(input_length, [], None)
    linear.append(train(network, training, validation, testing, 0.7e-3, 10, 15, 1e-5))
    print(f"{i}%")

# Print results
print(f"Relu: {round(sum(relu) / len(relu), 2)}, Sigmoid: {round(sum(sigmoid) / len(sigmoid), 2)}, Linear: {round(sum(linear) / len(linear), 2)}")


