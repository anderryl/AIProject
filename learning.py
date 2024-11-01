import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_iteration(dataloader, model, loss_fn, optimizer, batch_size, verbose=False):
    """
    Runs a single epoch of training
    :param dataloader: dataloader of training data
    :param model: model to train
    :param loss_fn: loss function
    :param optimizer: back-propogation optimizer
    :param batch_size: Number of samples in each back
    :param verbose: determines whether to print diagnostic data each epoch
    """
    size = len(dataloader.dataset)
    # Set the model to training mode
    model.train()

    # Loop through each batches in data loader
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # For fine-tuning purposes
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            if verbose: print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validation_iteration(dataloader, model, loss_fn, label, verbose=False):
    """
    Runs a single validation iteration without training the model
    :param dataloader: validation (or testing) data loader
    :param model: classifier model
    :param loss_fn: loss function
    :param label: label for printing diagnostic data
    :param verbose: determines whether to print diagnostic data
    :return: model accuracy
    """
    # Set the model to evaluation mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() to not compute gradients for backpropogation
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # Check each prediction in the batch
            for actual, predicted in zip(y.tolist(), pred.argmax(1).tolist()):
                if actual == predicted:
                    correct += 1

    # Compute the test statistics
    test_loss /= num_batches
    correct /= size
    if verbose: print(f"{label} Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100 * correct


def train(model, training, validation, testing, learning_rate, batch_size, epochs, l_two, verbose=False):
    """
    Trains a classifier model with given hyperparameters
    :param model: classifier model
    :param training: training dataset
    :param validation: validation dataset
    :param testing: testing dataset
    :param learning_rate: learning rate
    :param batch_size: number of samples per epoch
    :param epochs: number of epochs to run for
    :param l_two: L2 regularization weight
    :param verbose: determines whether to print disgnostic data
    :return: testing data accuracy
    """

    # Cross Entory Loss function which is optimized for training classifiers
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l_two)
    training_loader = DataLoader(training, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True)
    testing_loader = DataLoader(testing, batch_size=batch_size, shuffle=False)

    # Train and validate during each epoch
    for t in range(epochs):
        if verbose: print(f"Epoch {t + 1}\n-------------------------------")
        train_iteration(training_loader, model, loss_fn, optimizer, batch_size)
        validation_iteration(validation_loader, model, loss_fn, "Validation")

    # Return the accuracy over testing data
    return validation_iteration(testing_loader, model, loss_fn, "Testing")
