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
    plt.savefig('performance.pdf')
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
