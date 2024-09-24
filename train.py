import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.model import ImageClassifier
from utils.data_loader import data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
model = ImageClassifier()
model.to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

train_loader, val_loader = data_loader()

num_epochs = 25
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(tqdm(train_loader, desc = f"Epoch {epoch + 1}/{num_epochs}", ncols = 80)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}%")

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for i, data in enumerate(tqdm(val_loader, desc = "Validation", ncols = 80)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Validation Accuracy: {val_acc:.4f}%")

torch.save(model.state_dict(), "checkpoints/model.pt")