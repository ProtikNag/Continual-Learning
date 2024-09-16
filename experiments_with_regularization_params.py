import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from gcn import GCN
from mer import MER
from visualization import plot_impact_of_parameters
from utils import set_seed
from graph_utils import split_by_class
from tqdm import tqdm
import warnings
import numpy as np
import random

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


def evaluate_model(model, test_data_list):
    model.eval()
    total_correct = 0
    total_nodes = 0
    for test_data in test_data_list:
        # Use the test data directly (no need for mask)
        out = model(test_data.x, test_data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred == test_data.y).sum().item()
        total_correct += correct
        total_nodes += test_data.num_nodes
    return total_correct / total_nodes


def train_test_split(data, train_ratio=0.8):
    num_nodes = data.num_nodes
    indices = list(range(num_nodes))
    random.shuffle(indices)

    train_size = int(train_ratio * num_nodes)

    # Create boolean masks for train and test splits
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True

    # Assign the masks to the data object
    data.train_mask = train_mask
    data.test_mask = test_mask

    train_data = data.subgraph(train_mask)
    test_data = data.subgraph(test_mask)

    return train_data, test_data


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
beta_values = np.linspace(0.01, 0.9, 5)
gamma_values = np.linspace(0.01, 0.9, 5)

performance_results = []

for beta in tqdm(beta_values):
    for gamma in tqdm(gamma_values):
        print(f"Training MER with beta={beta}, gamma={gamma}")

        model_mer = GCN(in_channels=dataset.num_node_features, out_channels=dataset.num_classes)
        mer = MER(model=model_mer, memory_size=100, batch_size=64, lr=0.01, alpha=0.1, beta=beta, gamma=gamma)

        mer_performance = {int(i): [] for i in range(len(train_data_list))}

        for task_id, train_data in enumerate(train_data_list):
            task_loader = DataLoader([train_data])

            for epoch in range(70):
                for data in task_loader:
                    mer.train_step(data)

            # After training, evaluate on all tasks (using test data list)
            for eval_task_id in range(len(test_data_list)):
                acc = evaluate_model(model_mer, [test_data_list[eval_task_id]])
                mer_performance[task_id].append(acc)

        # Store results for the current beta and gamma
        avg_acc = np.mean([np.mean(accs) for accs in mer_performance.values()])
        performance_results.append((beta, gamma, avg_acc))

# Convert results to arrays for easy plotting
performance_results = np.array(performance_results)

plot_impact_of_parameters(performance_results)
