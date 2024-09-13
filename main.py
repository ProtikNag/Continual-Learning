# import torch
# import torch.nn as nn
# from torch_geometric.loader import DataLoader
# from torch_geometric.datasets import Planetoid
#
# from gcn import GCN
# from mer import MER
# from visualization import plot_loss_curve, plot_performance_histograms
# from graph_utils import split_by_class
# from tqdm import tqdm
#
# import warnings
#
# warnings.simplefilter(action='ignore', category=FutureWarning)
#
# # Load the Cora dataset
# dataset = Planetoid(root='data/Planetoid', name='Cora')
#
# # Define class splits for tasks
# class_splits = [
#     [0, 1],
#     [2],
#     [3],
#     [4, 5],
#     [6]
# ]
#
# # Create a task list by filtering nodes based on class
# task_data_list = []
# for class_split in class_splits:
#     task_data = split_by_class(dataset[0], class_split)
#     task_data_list.append(task_data)
#
# # See the data
# for i, task_data in enumerate(task_data_list):
#     print(f"Task {i + 1} has {task_data.num_nodes} nodes and {task_data.num_edges} edges")
#
#
# # Function to evaluate model on a list of tasks
# def evaluate_model(model, task_data_list):
#     model.eval()
#     total_correct = 0
#     total_nodes = 0
#     for task_data in task_data_list:
#         out = model(task_data.x, task_data.edge_index)
#         pred = out.argmax(dim=1)
#         correct = (pred == task_data.y).sum().item()
#         total_correct += correct
#         total_nodes += task_data.y.size(0)
#     return total_correct / total_nodes  # Return overall accuracy across tasks
#
#
# # Case 2: Naive Learning
# model_naive = GCN(in_channels=dataset.num_node_features, out_channels=dataset.num_classes)
# optimizer_naive = torch.optim.Adam(model_naive.parameters(), lr=0.01)
# criterion = nn.CrossEntropyLoss()
#
# # Track performance on all tasks after training on each task
# naive_learning_performance = {int(i): [] for i in range(len(task_data_list))}
#
# # Training the model on one task at a time
# for task_id, task_data in tqdm(enumerate(task_data_list)):
#     task_loader = DataLoader([task_data])
#
#     # Train on the current task
#     for epoch in range(50):
#         model_naive.train()
#         total_loss = 0
#         for data in task_loader:
#             optimizer_naive.zero_grad()
#             out = model_naive(data.x, data.edge_index)
#             loss = criterion(out, data.y)
#             loss.backward()
#             optimizer_naive.step()
#             total_loss += loss.item()
#
#     # After training, evaluate on all tasks
#     for task_id in range(len(task_data_list)):
#         acc = evaluate_model(model_naive, task_data_list[:task_id + 1])
#         naive_learning_performance[task_id].append(acc)
#
# # Case 3: MER
# model_mer = GCN(in_channels=dataset.num_node_features, out_channels=dataset.num_classes)
# mer = MER(model=model_mer, memory_size=200, batch_size=128, lr=0.01, alpha=0.1, beta=0.1, gamma=0.01)
#
# # Track performance on all tasks after training on each task
# mer_performance = {int(i): [] for i in range(len(task_data_list))}
#
# # Training on each task with MER
# for task_id, task_data in tqdm(enumerate(task_data_list)):
#     task_loader = DataLoader([task_data])
#
#     # Train on the current task with MER
#     for epoch in range(50):
#         total_loss = 0
#         for data in task_loader:
#             loss = mer.train_step(data)
#             total_loss += loss.item()
#
#     for task_id in range(len(task_data_list)):
#         acc = evaluate_model(model_mer, task_data_list[:task_id + 1])
#         mer_performance[task_id].append(acc)
#
# print("Naive Learning Performance:", naive_learning_performance)
# print("MER Performance:", mer_performance)
#
# plot_performance_histograms(naive_learning_performance, mer_performance, title='Performance Histograms')
#


import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from gcn import GCN
from mer import MER
from visualization import plot_loss_curve, plot_performance_histograms
from graph_utils import split_by_class
from tqdm import tqdm
import random
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

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


# Function to split the data into train and test for a given task
def train_test_split(data, train_ratio=0.8):
    num_nodes = data.num_nodes
    indices = list(range(num_nodes))
    random.shuffle(indices)

    train_size = int(train_ratio * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True

    data.train_mask = train_mask
    data.test_mask = test_mask

    return data


# Create a task list by filtering nodes based on class
task_data_list = []
for class_split in class_splits:
    task_data = split_by_class(dataset[0], class_split)
    task_data_list.append(task_data)

# See the data
for i, task_data in enumerate(task_data_list):
    print(f"Task {i + 1} has {task_data.num_nodes} nodes and {task_data.num_edges} edges")


# Function to evaluate model on the test set
def evaluate_model(model, task_data_list, mask_name='test_mask'):
    model.eval()
    total_correct = 0
    total_nodes = 0
    for task_data in task_data_list:
        mask = getattr(task_data, mask_name)
        subgraph_data = task_data.subgraph(mask)  # Extract subgraph based on mask
        out = model(subgraph_data.x, subgraph_data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred == subgraph_data.y).sum().item()
        total_correct += correct
        total_nodes += subgraph_data.num_nodes
    return total_correct / total_nodes  # Return accuracy across tasks


# Case 2: Naive Learning
model_naive = GCN(in_channels=dataset.num_node_features, out_channels=dataset.num_classes)
optimizer_naive = torch.optim.Adam(model_naive.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Track performance on all tasks after training on each task
naive_learning_performance = {int(i): [] for i in range(len(task_data_list))}

# Training the model on one task at a time
for task_id, task_data in tqdm(enumerate(task_data_list)):
    # Split train and test for each task
    task_data = train_test_split(task_data, train_ratio=0.8)

    # Train on the current task
    task_loader = DataLoader([task_data])
    for epoch in range(50):
        model_naive.train()
        total_loss = 0
        for data in task_loader:
            subgraph_data = data.subgraph(data.train_mask)  # Extract subgraph based on train_mask
            optimizer_naive.zero_grad()
            out = model_naive(subgraph_data.x, subgraph_data.edge_index)
            loss = criterion(out, subgraph_data.y)
            loss.backward()
            optimizer_naive.step()
            total_loss += loss.item()

    # After training, evaluate on all tasks (using test masks)
    for eval_task_id in range(len(task_data_list)):
        acc = evaluate_model(model_naive, task_data_list[:eval_task_id + 1], mask_name='test_mask')
        naive_learning_performance[eval_task_id].append(acc)

# Case 3: MER
model_mer = GCN(in_channels=dataset.num_node_features, out_channels=dataset.num_classes)
mer = MER(model=model_mer, memory_size=200, batch_size=128, lr=0.01, alpha=0.1, beta=0.1, gamma=0.01)

# Track performance on all tasks after training on each task
mer_performance = {int(i): [] for i in range(len(task_data_list))}

# Training on each task with MER
for task_id, task_data in tqdm(enumerate(task_data_list)):
    # Split train and test for each task
    task_data = train_test_split(task_data, train_ratio=0.8)

    task_loader = DataLoader([task_data])
    # Train on the current task with MER
    for epoch in range(50):
        total_loss = 0
        for data in task_loader:
            loss = mer.train_step(data)
            total_loss += loss.item()

    # After training, evaluate on all tasks (using test masks)
    for eval_task_id in range(len(task_data_list)):
        acc = evaluate_model(model_mer, task_data_list[:eval_task_id + 1], mask_name='test_mask')
        mer_performance[eval_task_id].append(acc)

# Display results
print("Naive Learning Performance:", naive_learning_performance)
print("MER Performance:", mer_performance)

plot_performance_histograms(naive_learning_performance, mer_performance, title='Performance Histograms')

