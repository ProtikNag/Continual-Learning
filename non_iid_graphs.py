import networkx as nx
import numpy as np
from collections import Counter
from scipy.stats import entropy
import random

# Parameters
num_nodes = 10
num_edges = 15
node_types_list = ['A', 'B', 'C']
edge_types_list = ['U', 'V', 'W', 'X', 'Y', 'Z']

G = nx.Graph()

for i in range(1, num_nodes + 1):
    node_type = random.choice(node_types_list)
    G.add_node(i, type=node_type)

for _ in range(num_edges):
    node1, node2 = random.sample(list(G.nodes()), 2)
    edge_type = random.choice(edge_types_list)
    G.add_edge(node1, node2, type=edge_type)

# Node Type Heterogeneity
node_types = [data['type'] for _, data in G.nodes(data=True)]
type_counts = Counter(node_types)
type_probabilities = np.array(list(type_counts.values())) / len(node_types)
node_type_heterogeneity = entropy(type_probabilities)

# Edge Type Heterogeneity
edge_types = [data['type'] for _, _, data in G.edges(data=True)]
edge_type_counts = Counter(edge_types)
edge_type_probabilities = np.array(list(edge_type_counts.values())) / len(edge_types)
edge_type_heterogeneity = entropy(edge_type_probabilities)

# Degree Distribution Heterogeneity
degrees = [deg for _, deg in G.degree()]
degree_counts = Counter(degrees)
degree_probabilities = np.array(list(degree_counts.values())) / len(degrees)
degree_distribution_heterogeneity = entropy(degree_probabilities)

print(f"Node Type Heterogeneity: {node_type_heterogeneity:.4f}")
print(f"Edge Type Heterogeneity: {edge_type_heterogeneity:.4f}")
print(f"Degree Distribution Heterogeneity: {degree_distribution_heterogeneity:.4f}")
