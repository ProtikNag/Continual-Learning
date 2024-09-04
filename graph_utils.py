import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data

from utils import NUM_NODES, NUM_FEATURES, NUM_EDGES

base_graph = torch.randn((NUM_NODES, NUM_FEATURES))
edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))


# Create synthetic graph data for node classification
def create_synthetic_graph(num_nodes, num_features, num_classes, dissimilarity=0.0):
    x = base_graph + dissimilarity * torch.randn((num_nodes, num_features))
    y = torch.randint(0, num_classes, (num_nodes,))
    return Data(x=x, edge_index=edge_index, y=y)


def plot_graph(G, node_labels, title):
    plt.figure(figsize=(8, 8))
    node_colors = [node_labels[i] for i in G.nodes()]
    nx.draw(G, node_color=node_colors, with_labels=True, cmap=plt.cm.rainbow, node_size=500, font_color='white')
    plt.title(title)
    plt.show()
