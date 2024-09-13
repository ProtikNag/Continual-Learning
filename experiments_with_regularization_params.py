import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from gcn import GCN
from mer import MER
from visualization import plot_loss_curve, plot_performance_histograms
from main import evaluate_model, train_test_split
from utils import set_seed
from graph_utils import split_by_class
from tqdm import tqdm
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
set_seed(77)

# Load the Cora dataset
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=T.NormalizeFeatures())

# Define class splits for tasks
class_splits = [
    [0, 1],
    [2],
    [3],
    [4, 5],
    [6]
]

# Create separate train and test data lists, split by class
train_data_list = []
test_data_list = []
for class_split in class_splits:
    task_data = split_by_class(dataset[0], class_split)

    # Split the task data into training and testing subsets
    train_data, test_data = train_test_split(task_data, train_ratio=0.8)

    # Append to the respective lists
    train_data_list.append(train_data)
    test_data_list.append(test_data)

# Grid search over beta and gamma
beta_values = np.linspace(0.001, 0.9, 5)  # Example: 5 values for beta
gamma_values = np.linspace(0.001, 0.9, 5)  # Example: 5 values for gamma

performance_results = []
loss_results = []

for beta in beta_values:
    for gamma in gamma_values:
        print(f"Training MER with beta={beta}, gamma={gamma}")

        model_mer = GCN(in_channels=dataset.num_node_features, out_channels=dataset.num_classes)
        mer = MER(model=model_mer, memory_size=100, batch_size=64, lr=0.01, alpha=0.1, beta=beta, gamma=gamma)

        mer_performance = {int(i): [] for i in range(len(train_data_list))}
        total_loss_over_tasks = 0

        for task_id, train_data in tqdm(enumerate(train_data_list)):
            task_loader = DataLoader([train_data])

            for epoch in range(70):
                total_loss = 0
                for data in task_loader:
                    loss = mer.train_step(data)
                    total_loss += loss.item()
                total_loss_over_tasks += total_loss

            # After training, evaluate on all tasks (using test data list)
            for eval_task_id in range(len(test_data_list)):
                acc = evaluate_model(model_mer, [test_data_list[eval_task_id]])
                mer_performance[task_id].append(acc)

        # Store results for the current beta and gamma
        avg_acc = np.mean([np.mean(accs) for accs in mer_performance.values()])
        performance_results.append((beta, gamma, avg_acc))
        loss_results.append((beta, gamma, total_loss_over_tasks))

# Convert results to arrays for easy plotting
performance_results = np.array(performance_results)
loss_results = np.array(loss_results)

# Plotting the performance as a function of beta and gamma
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Performance plot
scatter = ax[0].scatter(performance_results[:, 0], performance_results[:, 1], c=performance_results[:, 2],
                        cmap='viridis')
ax[0].set_title('MER Performance (Accuracy) vs Beta, Gamma')
ax[0].set_xlabel('Beta')
ax[0].set_ylabel('Gamma')
fig.colorbar(scatter, ax=ax[0])

# Loss plot
scatter_loss = ax[1].scatter(loss_results[:, 0], loss_results[:, 1], c=loss_results[:, 2], cmap='plasma')
ax[1].set_title('MER Loss vs Beta, Gamma')
ax[1].set_xlabel('Beta')
ax[1].set_ylabel('Gamma')
fig.colorbar(scatter_loss, ax=ax[1])

plt.show()
