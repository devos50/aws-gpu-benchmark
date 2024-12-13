import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop
from torch.utils.data import DataLoader
from datasets import load_dataset

from args import get_args, SUPPORTED_MODELS

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()) if cuda_available else "cpu"
print("Using device: %s (GPU: %s)" % (device, gpu_name))

def train(model, model_name: str, dataloader, criterion, optimizer, log=True):
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
        if "vit" in model_name:
            outputs = outputs.logits
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
                f.write(f"{gpu_name},{model_name},cifar10,{args.batch_size},{step_time:.4f}\n")

    accuracy = 100. * correct / total
    return running_loss / len(dataloader), accuracy

def init_data_dir():
    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists("data/model_load_times.csv"):
        with open("data/model_load_times.csv", "w") as f:
            f.write("gpu,model,dataset,load_time\n")

    if not os.path.exists("data/local_steps_time.csv"):
        with open("data/local_steps_time.csv", "w") as f:
            f.write("gpu,model,dataset,batch_size,step_time\n")

def get_model(model_name):
    if model_name == "resnet18":
        from torchvision.models import resnet18
        return resnet18(num_classes=10)
    elif model_name == "resnet34":
        from torchvision.models import resnet34
        return resnet34(num_classes=10)
    elif model_name == "resnet50":
        from torchvision.models import resnet50
        return resnet50(num_classes=10)
    elif model_name == "resnet101":
        from torchvision.models import resnet101
        return resnet101(num_classes=10)
    elif model_name == "resnet152":
        from torchvision.models import resnet152
        return resnet152(num_classes=10)
    elif "vit" in model_name:
        from transformers import ViTForImageClassification
        return ViTForImageClassification.from_pretrained('google/%s' % model_name)

def get_transformation(model_name):
    if "vit" in model_name:
        return Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ])
    else:
        return Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

def benchmark(args):
    init_data_dir()

    # Load CIFAR-10 dataset using Hugging Face
    data = load_dataset("cifar10")

    # Prepare datasets and dataloaders
    def preprocess_function(examples, transform):
        images = [transform(image.convert("RGB")) for image in examples["img"]]
        labels = examples["label"]
        return {"images": images, "labels": labels}

    models_to_test = SUPPORTED_MODELS if args.test_all else [args.model]
    for model_name in models_to_test:
        print("Testing model: %s" % model_name)
        model = get_model(model_name)

        transform_train = get_transformation(model_name)
        train_dataset = data["train"].with_transform(lambda x: preprocess_function(x, transform_train))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x, drop_last=True)

        start_time = time.time()
        model = model.to(device)
        elapsed_time = time.time() - start_time
        print("Model loaded to device in %.2f seconds." % elapsed_time)
        with open("data/model_load_times.csv", "a") as f:
            f.write(f"{gpu_name},{model_name},cifar10,{elapsed_time:.4f}\n")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        # First, we perform a single epoch to warm up the GPU and caches
        print("Running warm-up epoch")
        _ = train(model, model_name, train_loader, criterion, optimizer, log=False)

        # Main training loop
        for epoch in range(args.num_epochs):
            train_loss, train_accuracy = train(model, model_name, train_loader, criterion, optimizer)

            print(f"Epoch {epoch+1}/{args.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    print("Benchmark completed.")

if __name__ == "__main__":
    args = get_args()
    benchmark(args)
