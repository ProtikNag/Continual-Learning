import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from gcn import GCN
from mer import MER
from visualization import plot_loss_curve, plot_performance_histograms
from utils import set_seed
from graph_utils import split_by_class
from tqdm import tqdm
import random
import warnings

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


# Function to split the data into train and test for a given task
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

# See the data
for i, (train_data, test_data) in enumerate(zip(train_data_list, test_data_list)):
    print(f"Task {i + 1}: Train Data has {train_data.num_nodes} nodes, Test Data has {test_data.num_nodes} nodes")


# Function to evaluate model on the test set
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


# Case 2: Naive Learning
model_naive = GCN(in_channels=dataset.num_node_features, out_channels=dataset.num_classes)
optimizer_naive = torch.optim.Adam(model_naive.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Track performance on all tasks after training on each task
naive_learning_performance = {int(i): [] for i in range(len(train_data_list))}

# Training the model on one task at a time
for task_id, train_data in tqdm(enumerate(train_data_list)):
    # No need for further split, use train_data directly
    task_loader = DataLoader([train_data])

    # Train on the current task
    for epoch in range(70):
        model_naive.train()
        total_loss = 0
        for data in task_loader:
            optimizer_naive.zero_grad()
            out = model_naive(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer_naive.step()
            total_loss += loss.item()

    # After training, evaluate on all tasks
    for eval_task_id in range(len(test_data_list)):
        acc = evaluate_model(model_naive, [test_data_list[eval_task_id]])
        naive_learning_performance[task_id].append(acc)

# Case 3: MER
model_mer = GCN(in_channels=dataset.num_node_features, out_channels=dataset.num_classes)
mer = MER(model=model_mer, memory_size=100, batch_size=64, lr=0.01, alpha=0.1, beta=0.9, gamma=0.9)

# Track performance on all tasks after training on each task
mer_performance = {int(i): [] for i in range(len(train_data_list))}

# Training on each task with MER
for task_id, train_data in tqdm(enumerate(train_data_list)):
    task_loader = DataLoader([train_data])

    # Train on the current task with MER
    for epoch in range(70):
        total_loss = 0
        for data in task_loader:
            loss = mer.train_step(data)
            total_loss += loss.item()

    # After training, evaluate on all tasks (using test data list)
    for eval_task_id in range(len(test_data_list)):
        acc = evaluate_model(model_mer, [test_data_list[eval_task_id]])
        mer_performance[task_id].append(acc)

# Case 1: All Data Together
# Merge all training data
all_train_data = torch_geometric.data.Data(
    x=torch.cat([data.x for data in train_data_list], dim=0),
    edge_index=torch.cat([data.edge_index for data in train_data_list], dim=1),
    y=torch.cat([data.y for data in train_data_list], dim=0)
)

# Merge all test data
all_test_data = torch_geometric.data.Data(
    x=torch.cat([data.x for data in test_data_list], dim=0),
    edge_index=torch.cat([data.edge_index for data in test_data_list], dim=1),
    y=torch.cat([data.y for data in test_data_list], dim=0)
)

# Create a DataLoader for the merged train data
train_loader = DataLoader([all_train_data], batch_size=1, shuffle=True)

# Case 1: Train the model on the combined data
model_all_data = GCN(in_channels=dataset.num_node_features, out_channels=dataset.num_classes)
optimizer_all_data = torch.optim.Adam(model_all_data.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Train the model on all data together
for epoch in range(500):
    model_all_data.train()
    total_loss = 0
    for data in train_loader:
        optimizer_all_data.zero_grad()
        out = model_all_data(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer_all_data.step()
        total_loss += loss.item()

# Evaluate the model on all test data together
all_data_accuracy = evaluate_model(model_all_data, [all_test_data])

print(f"Accuracy on all test data together: {all_data_accuracy}")

plot_performance_histograms(naive_learning_performance, mer_performance, title='Performance Histograms')
