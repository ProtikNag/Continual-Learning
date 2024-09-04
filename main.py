import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

from graph_utils import create_synthetic_graph, plot_graph
from utils import NUM_CLASSES, NUM_NODES, NUM_FEATURES, NUM_EDGES, NUM_TASKS
from mer import MER
from visualization import plot_loss_curve

from gcn import GCN

# Case 1: Showing Model All Data Together

# Generate tasks with increasing dissimilarity
task_data_list = [create_synthetic_graph(NUM_NODES, NUM_FEATURES, NUM_CLASSES, dissimilarity=i * 0.5) for i in range(NUM_TASKS)]
task_data_loader = DataLoader(task_data_list, batch_size=8, shuffle=True)

model = GCN(in_channels=NUM_FEATURES, out_channels=NUM_CLASSES)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

loss_for_all_data_together = []

for epoch in range(100):
    total_loss = 0
    for data in task_data_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss_for_all_data_together.append(total_loss)

# Case 2: Naive Learning
task_data_list = [create_synthetic_graph(NUM_NODES, NUM_FEATURES, NUM_CLASSES, dissimilarity=i * 0.5) for i in range(NUM_TASKS)]

model = GCN(in_channels=NUM_FEATURES, out_channels=NUM_CLASSES)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training the model on one task at a time
loss_for_naive_learning = []

for i, task_data in enumerate(task_data_list):
    task_data_loader = DataLoader([task_data], batch_size=1, shuffle=True)
    for epoch in range(100):
        total_loss = 0
        for data in task_data_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_for_naive_learning.append(total_loss)


# Case 3: MER

task_data_list = [create_synthetic_graph(NUM_NODES, NUM_FEATURES, NUM_CLASSES, dissimilarity=i * 0.5) for i in range(NUM_TASKS)]

model = GCN(in_channels=NUM_FEATURES, out_channels=NUM_CLASSES)
mer = MER(model=model, memory_size=50, batch_size=10, lr=0.01, alpha=0.1, beta=0.01, gamma=0.1)

loss_for_mer = []

for i, task_data in enumerate(task_data_list):
    task_data_loader = DataLoader([task_data], batch_size=1, shuffle=True)
    for epoch in range(100):
        total_loss = 0
        for data in task_data_loader:
            loss = mer.train_step(data)
            total_loss += loss.item()
        loss_for_mer.append(total_loss)


plot_loss_curve(loss_for_all_data_together, title='All Data Together', case="All Data Together")
plot_loss_curve(loss_for_naive_learning, title='Naive Learning', case="Naive Learning")
plot_loss_curve(loss_for_mer, title='MER', case="MER")
