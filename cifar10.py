import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop
from torch.utils.data import DataLoader
from datasets import load_dataset

from args import get_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)

def train(model, dataloader, criterion, optimizer, log=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    steps = 0

    epoch_start_time = time.time()
    local_step_times = []
    for batch in dataloader:
        start_time = time.time()
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
        if log:
            local_step_times.append(elapsed_time)

        # Clear GPU memory
        del images, labels, outputs
        torch.cuda.empty_cache()

    elapsed_epoch_time = time.time() - epoch_start_time
    print("Running one epoch (%d steps) took %.2f seconds." % (steps, elapsed_epoch_time))

    if log:
        with open("data/local_steps_time.csv", "a") as f:
            for step_time in local_step_times:
                f.write(f"{args.model},cifar10,{step_time:.4f}\n")

    accuracy = 100. * correct / total
    return running_loss / len(dataloader), accuracy

def init_data_dir():
    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists("data/model_load_times.csv"):
        with open("data/model_load_times.csv", "w") as f:
            f.write("model,dataset,load_time\n")

    if not os.path.exists("data/local_steps_time.csv"):
        with open("data/local_steps_time.csv", "w") as f:
            f.write("model,dataset,step_time\n")

def benchmark(args):
    init_data_dir()

    # Load CIFAR-10 dataset using Hugging Face
    data = load_dataset("cifar10")

    # Data preprocessing
    transform_train = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Prepare datasets and dataloaders
    def preprocess_function(examples, transform):
        images = [transform(image.convert("RGB")) for image in examples["img"]]
        labels = examples["label"]
        return {"images": images, "labels": labels}

    train_dataset = data["train"].with_transform(lambda x: preprocess_function(x, transform_train))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x, drop_last=True)

    # Define model, loss, and optimizer
    model = resnet18(num_classes=10)

    start_time = time.time()
    model = model.to(device)
    elapsed_time = time.time() - start_time
    print("Model loaded to device in %.2f seconds." % elapsed_time)
    with open("data/model_load_times.csv", "a") as f:
        f.write(f"{args.model},cifar10,{elapsed_time:.4f}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # First, we perform a single epoch to warm up the GPU and caches
    print("Running warm-up epoch")
    _ = train(model, train_loader, criterion, optimizer, log=False)

    # Main training loop
    for epoch in range(args.num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)

        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    print("Benchmark completed.")

if __name__ == "__main__":
    args = get_args()
    benchmark(args)
