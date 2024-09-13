import matplotlib.pyplot as plt
import networkx as nx
import torch


# Function to split data by class (each task gets different classes)
def split_by_class(data, classes):
    mask = torch.isin(data.y, torch.tensor(classes))
    return data.clone().subgraph(mask.nonzero(as_tuple=True)[0])


def plot_graph(G, node_labels, title):
    plt.figure(figsize=(8, 8))
    node_colors = [node_labels[i] for i in G.nodes()]
    nx.draw(G, node_color=node_colors, with_labels=True, cmap=plt.cm.rainbow, node_size=500, font_color='white')
    plt.title(title)
    plt.show()
