# Script to benchmark the time it takes to serialize a model to disk and load it from the disk
import os
import time

from models import get_model

import torch


# MODELS_TO_TEST = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
#                   "efficientnet-b7",
#                   "mobilenet_v3_large",
#                   "vit-base-patch16-224", "vit-large-patch16-224",
#                   "bert-base-uncased",
#                   "dense121", "dense169", "dense201", "dense161"]

MODELS_TO_TEST = ["gpt2"]

MODEL_PATH = os.path.join("data", "model.pt")
TIME_FILE_PATH = os.path.join("data", "serialization_times.csv")


def init_data_dir():
    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists(TIME_FILE_PATH):
        with open(TIME_FILE_PATH, "w") as f:
            f.write("model,model_size,serialization_time,deserialization_time\n")


def benchmark_serialization_speed(model_name):
    print(f"Testing serialization speed for model {model_name}...")

    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)    

    for _ in range(10):
        model = get_model(model_name, "cifar10")

        # Serialize the model
        start_time = time.time()
        model_data = torch.save(model.state_dict(), MODEL_PATH)
        serialize_time = time.time() - start_time
        print(f"Model serialized. Time taken: {serialize_time:.2f} seconds")

        serialized_size = os.path.getsize(MODEL_PATH)

        # Deserialize the model
        start_time = time.time()
        model = get_model(model_name, "cifar10")
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        deserialize_time = time.time() - start_time
        print(f"Model deserialized. Time taken: {deserialize_time:.2f} seconds")

        # Log serialization and deserialization times
        with open(TIME_FILE_PATH, "a") as f:
            f.write(f"{model_name},{serialized_size},{serialize_time:.4f},{deserialize_time:.4f}\n")

        del model
        time.sleep(1)


if __name__ == "__main__":
    init_data_dir()
    for model_name in MODELS_TO_TEST:
        benchmark_serialization_speed(model_name)
