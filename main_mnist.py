import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mer import MER
from visualization import plot_loss_curve
from cnn import CNN
from tqdm import tqdm

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Prepare DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Case 1: All Data Together
model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

loss_for_all_data_together = []

for epoch in tqdm(range(10)):  # Reduce epoch count for faster runs
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss_for_all_data_together.append(total_loss)

plot_loss_curve(loss_for_all_data_together, title='All Data Together', case="MNIST All Data Together")

# Case 2: Naive Learning - Assume task-based partition (e.g., different digits as tasks)
naive_loss = []
for task in tqdm(range(10)):  # 10 tasks for each digit
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    task_data = [(x, y) for x, y in train_dataset if y == task]  # Filter data per digit
    task_loader = DataLoader(task_data, batch_size=64, shuffle=True)

    for epoch in range(10):
        total_loss = 0
        for data, target in task_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        naive_loss.append(total_loss)

plot_loss_curve(naive_loss, title='Naive Learning', case="MNIST Naive Learning")

# Case 3: Using MER
model = CNN()
mer = MER(model=model, memory_size=100, batch_size=64, lr=0.01, alpha=0.1, beta=0.01, gamma=0.01)

mer_loss = []

for task in tqdm(range(10)):  # Again, assume tasks as different digits
    task_data = [(x, y) for x, y in train_dataset if y == task]
    task_loader = DataLoader(task_data, batch_size=64, shuffle=True)

    for epoch in range(10):
        total_loss = 0
        for data, target in task_loader:
            loss = mer.train_step(data, target)
            total_loss += loss.item()
        mer_loss.append(total_loss)

plot_loss_curve(mer_loss, title='MER', case="MNIST MER")
