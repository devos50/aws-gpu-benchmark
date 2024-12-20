# Script to benchmark the time it takes to serialize a model to disk and load it from the disk
import os
import time

from models import get_model

import torch


MODELS_TO_TEST = ["resnet18", "resnet50", "efficientnet-b7", "vit-base-patch16-224"]
MODEL_PATH = os.path.join("data", "model.pt")


def benchmark_serialization_speed(model_name):
    print(f"Testing serialization speed for model {model_name}...")

    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    # Load the model
    print(f"Loading model {model_name}...")
    model = get_model(model_name, "cifar10")

    # Serialize the model
    start_time = time.time()
    model_data = torch.save(model.state_dict(), MODEL_PATH)
    serialize_time = time.time() - start_time
    print(f"Model serialized. Time taken: {serialize_time:.2f} seconds")

    # Deserialize the model
    start_time = time.time()
    model = get_model(model_name, "cifar10")
    model.load_state_dict(torch.load(MODEL_PATH))
    deserialize_time = time.time() - start_time
    print(f"Model deserialized. Time taken: {deserialize_time:.2f} seconds")


if __name__ == "__main__":
    for model_name in MODELS_TO_TEST:
        benchmark_serialization_speed(model_name)
