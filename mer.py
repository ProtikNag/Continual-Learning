import random

import torch.nn as nn
import torch.optim as optim


class MER:
    def __init__(self, model, memory_size=100, batch_size=32, lr=0.01, alpha=0.1, beta=0.01, gamma=0.01):
        self.model = model
        self.memory = []
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def update_memory(self, data):
        if len(self.memory) < self.memory_size:
            self.memory.append(data)
        else:
            idx = random.randint(0, len(self.memory) - 1)
            self.memory[idx] = data

    def sample_memory(self):
        return random.sample(self.memory, min(len(self.memory), self.batch_size))

    def train_step(self, current_data):
        memory_samples = self.sample_memory()
        s = len(memory_samples)  # Number of sampled batches

        # Save initial model parameters for meta-updates
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}

        # Within-batch Reptile meta-updates
        for data in memory_samples:
            x, edge_index, y = data.x, data.edge_index, data.y
            self.optimizer.zero_grad()
            out = self.model(x, edge_index)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            self.optimizer.step()

            # Meta-update within the batch
            updated_params = {name: param.clone() for name, param in self.model.named_parameters()}
            for name, param in self.model.named_parameters():
                param.data = initial_params[name] + self.beta * (updated_params[name] - initial_params[name])

        # Across-batch Reptile meta-update
        final_params = {name: param.clone() for name, param in self.model.named_parameters()}
        for name, param in self.model.named_parameters():
            param.data = initial_params[name] + self.gamma * (final_params[name] - initial_params[name])

        # Now train on the current data
        self.optimizer.zero_grad()
        x, edge_index, y = current_data.x, current_data.edge_index, current_data.y
        out = self.model(x, edge_index)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        self.optimizer.step()

        # Update memory with the current data
        self.update_memory(current_data)

        return loss
