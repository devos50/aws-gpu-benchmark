import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop
from torch.utils.data import DataLoader
from datasets import load_dataset

# Configuration
batch_size = 128
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device: %s" % device)

# Load CIFAR-10 dataset using Hugging Face
data = load_dataset("cifar10")

# Data preprocessing
transform_train = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Prepare datasets and dataloaders
def preprocess_function(examples, transform):
    images = [transform(image.convert("RGB")) for image in examples["img"]]
    labels = examples["label"]
    return {"images": images, "labels": labels}

train_dataset = data["train"].with_transform(lambda x: preprocess_function(x, transform_train))
test_dataset = data["test"].with_transform(lambda x: preprocess_function(x, transform_test))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x, drop_last=True)

# Define model, loss, and optimizer
model = resnet18(num_classes=10)

start_time = time.time()
model = model.to(device)
elapsed_time = time.time() - start_time
print("Model loaded to device in %.2f seconds." % elapsed_time)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Training loop
def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    steps = 0

    start_time = time.time()
    for batch in dataloader:
        images = torch.stack([item["images"] for item in batch]).to(device)
        labels = torch.tensor([item["labels"] for item in batch]).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        steps += 1

    elapsed_time = time.time() - start_time
    print("Running %d steps took %.2f seconds." % (steps, elapsed_time))

    accuracy = 100. * correct / total
    return running_loss / len(dataloader), accuracy

# Main script
for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

# Benchmarking the speed
import time

start_time = time.time()
train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
elapsed_time = time.time() - start_time
print(f"Training took {elapsed_time:.2f} seconds for one epoch.")
