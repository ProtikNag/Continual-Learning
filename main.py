import torch
from torch_geometric.data import Data, DataLoader

from gcn import GCN
from mer import MER

def create_synthetic_graph(num_nodes, num_features, num_classes):
    x = torch.randn((num_nodes, num_features))
    edge_index = torch.randint(0, num_nodes, (2, 200))
    y = torch.randint(0, num_classes, (num_nodes,))
    return Data(x=x, edge_index=edge_index, y=y)

task_data_list = [create_synthetic_graph(100, 16, 10) for _ in range(10)]
data_loader = DataLoader(task_data_list, batch_size=1)

# Initialize model and MER
model = GCN(in_channels=16, out_channels=10)
mer = MER(model=model, memory_size=50, batch_size=10, lr=0.01, alpha=0.1, beta=0.01)

# Train model with MER on different tasks
mer.train(data_loader, epochs=10)