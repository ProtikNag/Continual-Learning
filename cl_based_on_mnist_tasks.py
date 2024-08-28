import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from mer import MER
from gcn import GCN
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

# Download the MNIST dataset
mnist_train = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='../data', train=False, download=True, transform=transform)


def split_mnist_by_digits(dataset, digits):
    indices = [i for i, (x, y) in enumerate(dataset) if y in digits]
    return Subset(dataset, indices)


# Task 1: Digits 0-4, Task 2: Digits 5-9
task1_train = split_mnist_by_digits(mnist_train, digits=[0, 1, 2, 3, 4])
task2_train = split_mnist_by_digits(mnist_train, digits=[5, 6, 7, 8, 9])
task1_test = split_mnist_by_digits(mnist_test, digits=[0, 1, 2, 3, 4])
task2_test = split_mnist_by_digits(mnist_test, digits=[5, 6, 7, 8, 9])

# DataLoader for each task
task1_loader = DataLoader(task1_train, batch_size=64, shuffle=True)
task2_loader = DataLoader(task2_train, batch_size=64, shuffle=True)
task1_test_loader = DataLoader(task1_test, batch_size=1000, shuffle=False)
task2_test_loader = DataLoader(task2_test, batch_size=1000, shuffle=False)

model = GCN(in_channels=16, out_channels=10)
mer = MER(model=model, memory_size=50, batch_size=10, lr=0.01, alpha=0.1, beta=0.01)

# Initialize a dictionary to store losses for each task
losses = {
    'Task 1': [],
    'Task 2': []
}


def plot_performance(losses, title):
    plt.figure(figsize=(10, 6))
    for label, loss in losses.items():
        plt.plot(loss, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig('performance.pdf')
    plt.show()


# Train on each task sequentially
for task_label, loader in zip(losses.keys(), [task1_loader, task2_loader]):
    for data in loader:
        epoch_losses = []
        for epoch in range(10):  # Adjust the number of epochs as necessary
            loss = mer.train_step(data)
            epoch_losses.append(loss.item())
        losses[task_label] = epoch_losses

# Plot the performance
plot_performance(losses, 'MER Performance on MNIST Tasks')
