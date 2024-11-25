# Documentation statement: heavily used and modified code from pytorch's neural network documentation section

from expression import Exp
from loading import stats, create_datasets
from network import Net
import numpy as np
from learning import train
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind

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
    sigmoid.append(train(network, training, validation, testing, 0.9e-3, 10, 15, 1e-3))
    # Linear
    network = Net(input_length, [], None)
    linear.append(train(network, training, validation, testing, 0.9e-3, 10, 15, 1e-3))
    print(f"{i}%")

# Print results
print(f"Relu: {round(sum(relu) / len(relu), 2)}, Sigmoid: {round(sum(sigmoid) / len(sigmoid), 2)}, Linear: {round(sum(linear) / len(linear), 2)}")

results = pd.DataFrame({
    "ReLU": relu,
    "Sigmoid": sigmoid,
    "Linear": linear
})

print(results.describe)




# ANOVA test across all models
anova_result = f_oneway(results["ReLU"], results["Sigmoid"], results["Linear"])
print(f"ANOVA p-value: {anova_result.pvalue}")
if (anova_result.pvalue > 0.05) :
    print(f"No statistical difference among models")
else :
    print (f"Statistically significant difference among models")

print("\n")
# Pairwise t-tests
relu_sigmoid = ttest_ind(results["ReLU"], results["Sigmoid"])
relu_linear = ttest_ind(results["ReLU"], results["Linear"])
sigmoid_linear = ttest_ind(results["Sigmoid"], results["Linear"])

print(f"ReLU vs Sigmoid p-value: {relu_sigmoid.pvalue}")
if (relu_sigmoid.pvalue > 0.05) :
    print(f"No statistical difference between ReLU and Sigmoid ")
else :
    print (f"Statistically significant difference between ReLU and Sigmoid")
print("\n")

print(f"ReLU vs Linear p-value: {relu_linear.pvalue}")
if (relu_linear.pvalue > 0.05) :
    print(f"No statistical difference between ReLU and Linear ")
else :
    print (f"Statistically significant difference between ReLU and Linear")
print("\n")

print(f"Sigmoid vs Linear p-value: {sigmoid_linear.pvalue}")
if (sigmoid_linear.pvalue > 0.05) :
    print(f"No statistical difference between Sigmoid and Linear ")
else :
    print (f"Statistically significant difference between Sigmoid and Linear")
print("\n")

# Boxplot for comparison
results.boxplot()
plt.title("Accuracy Comparison Across Activation Functions")
plt.ylabel("Accuracy (%)")
plt.show()

# Histogram for each model
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# Plot ReLU
axes[0].hist(results["ReLU"], bins=20, color="blue", alpha=0.7)
axes[0].set_title("ReLU Accuracy Distribution")
axes[0].set_xlabel("Accuracy (%)")
axes[0].set_ylabel("Frequency")

# Plot Sigmoid
axes[1].hist(results["Sigmoid"], bins=20, color="orange", alpha=0.7)
axes[1].set_title("Sigmoid Accuracy Distribution")
axes[1].set_xlabel("Accuracy (%)")

# Plot Linear
axes[2].hist(results["Linear"], bins=20, color="green", alpha=0.7)
axes[2].set_title("Linear Accuracy Distribution")
axes[2].set_xlabel("Accuracy (%)")

# Adjust layout
plt.tight_layout()
plt.suptitle("Accuracy Distributions by Activation Function", y=1.02)
plt.show()

# Rolling mean over iterations
rolling_means = results.rolling(window=10).mean()

plt.plot(rolling_means.index, rolling_means["ReLU"], label="ReLU", color="blue")
plt.plot(rolling_means.index, rolling_means["Sigmoid"], label="Sigmoid", color="orange")
plt.plot(rolling_means.index, rolling_means["Linear"], label="Linear", color="green")

plt.legend(title="Activation Functions")
plt.title("Rolling Mean of Model Accuracy Over Iterations")
plt.xlabel("Iteration (Batches)")
plt.ylabel("Rolling Mean Accuracy (%)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
