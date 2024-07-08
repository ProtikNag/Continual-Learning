import numpy as np
import torch

import torch
from torch_geometric.data import Data, DataLoader

def create_synthetic_graph(num_nodes, num_features, num_classes, dissimilarity=0.0):
    x = torch.randn((num_nodes, num_features)) + dissimilarity * torch.randn((num_nodes, num_features))
    edge_index = torch.randint(0, num_nodes, (2, 200))
    y = torch.randint(0, num_classes, (num_nodes,))
    return Data(x=x, edge_index=edge_index, y=y)

# Create tasks with varying dissimilarity
task_data_list_dissimilar = [create_synthetic_graph(100, 16, 10, dissimilarity=i*0.5) for i in range(10)]
data_loader_dissimilar = DataLoader(task_data_list_dissimilar, batch_size=1)

# Varying number of tasks
task_data_list_few = [create_synthetic_graph(100, 16, 10) for _ in range(5)]
task_data_list_many = [create_synthetic_graph(100, 16, 10) for _ in range(20)]

data_loader_few = DataLoader(task_data_list_few, batch_size=1)
data_loader_many = DataLoader(task_data_list_many, batch_size=1)

# Create IID data
task_data_list_iid = [create_synthetic_graph(100, 16, 10) for _ in range(10)]
data_loader_iid = DataLoader(task_data_list_iid, batch_size=1)

# Create Non-IID data by shuffling the labels
def create_non_iid_data(data_list):
    non_iid_data_list = []
    for data in data_list:
        shuffled_y = data.y[torch.randperm(data.y.size(0))]
        non_iid_data_list.append(Data(x=data.x, edge_index=data.edge_index, y=shuffled_y))
    return non_iid_data_list

task_data_list_non_iid = create_non_iid_data(task_data_list_iid)
data_loader_non_iid = DataLoader(task_data_list_non_iid, batch_size=1)


import matplotlib.pyplot as plt

from gcn import GCN
from mer import MER

def plot_performance(losses, title):
    plt.figure(figsize=(10, 6))
    for label, loss in losses.items():
        plt.plot(loss, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()

# Training with different settings
losses = {
    'Dissimilar Tasks': [],
    'Few Tasks': [],
    'Many Tasks': [],
    'IID Data': [],
    'Non-IID Data': []
}

# Initialize model and MER for each experiment
model = GCN(in_channels=16, out_channels=10)
mer = MER(model=model, memory_size=50, batch_size=10, lr=0.01, alpha=0.1, beta=0.01)

# Train and collect losses for each setting
for label, loader in zip(losses.keys(), [data_loader_dissimilar, data_loader_few, data_loader_many, data_loader_iid, data_loader_non_iid]):
    for epoch in range(10):
        epoch_loss = 0
        for data in loader:
            loss = mer.train_step(data)
            epoch_loss += loss.item()
        losses[label].append(epoch_loss / len(loader))

# Plot the performance
plot_performance(losses, 'MER Performance Under Different Conditions')
