import torch
import torch.nn as nn


# Define a simple CNN for MNIST classification
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 24 * 24, 128)  # Corrected input dimensions
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Output shape: (batch_size, 32, 26, 26)
        x = torch.relu(self.conv2(x))  # Output shape: (batch_size, 64, 24, 24)
        x = x.view(x.size(0), -1)      # Flatten to (batch_size, 64 * 24 * 24)
        x = torch.relu(self.fc1(x))    # Pass through fully connected layer
        x = self.fc2(x)                # Final output
        return x