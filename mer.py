import torch.nn as nn
import torch.optim as optim
import random


class MER:
    def __init__(self, model, memory_size=100, batch_size=32, lr=0.01, alpha=0.1, beta=0.01):
        self.model = model
        self.memory = []
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.alpha = alpha
        self.beta = beta

    def update_memory(self, data):
        if len(self.memory) < self.memory_size:
            self.memory.append(data)
        else:
            idx = random.randint(0, len(self.memory) - 1)
            self.memory[idx] = data

    def sample_memory(self):
        return random.sample(self.memory, min(len(self.memory), self.batch_size))

    def train_step(self, current_data):
        # Sample memory
        memory_samples = self.sample_memory()

        # Within-batch Reptile meta-update
        for data in memory_samples:
            x, edge_index, y = data.x, data.edge_index, data.y
            self.optimizer.zero_grad()
            out = self.model(x, edge_index)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            self.optimizer.step()

        # Reptile meta-update
        self.optimizer.zero_grad()
        x, edge_index, y = current_data.x, current_data.edge_index, current_data.y
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}

        out = self.model(x, edge_index)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        self.optimizer.step()

        for name, param in self.model.named_parameters():
            param.data = initial_params[name] + self.beta * (param.data - initial_params[name])

        # Update memory
        self.update_memory(current_data)
        return loss

    def train(self, data_loader, epochs=10):
        for epoch in range(epochs):
            for data in data_loader:
                loss = self.train_step(data)
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')