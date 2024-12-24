# Script to benchmark the time it takes to serialize a model to disk and load it from the disk
import argparse
import os
import time

from models import get_model

import torch


MODELS_TO_TEST = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                  "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3", "efficientnet-b4", "efficientnet-b5", "efficientnet-b6", "efficientnet-b7",
                  "convnext-tiny", "convnext-small", "convnext-base", "convnext-large",
                  "mobilenet_v3_large",
                  "vit-base-patch16-224", "vit-large-patch16-224",
                  "densenet121", "densenet169", "densenet201", "densenet161",
                  "clip-vit-base-patch32", "clip-vit-large-patch14",
                  "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
                  "whisper-small",
                  "bert-base-uncased",
                  "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
                  "roberta-base", "roberta-large",
                  "distilbert-base-uncased", "distilbert-base-uncased-distilled-squad",
                  "albert-base-v2", "albert-large-v2", "albert-xlarge-v2", "albert-xxlarge-v2",
                  "t5-small", "t5-base", "t5-large",
                  ]

TIME_FILE_PATH = os.path.join("data", "serialization_times.csv")


def init_data_dir():
    if not os.path.exists("data"):
        os.makedirs("data")

    with open(TIME_FILE_PATH, "w") as f:
        f.write("instance,model,model_size,storage,serialization_time,deserialization_time,to_gpu_time,from_gpu_time\n")


def benchmark_serialization_speed(model_name, storage_backend, instance):
    print(f"Testing serialization speed for model {model_name} (backend: {storage_backend})...")

    if storage_backend == "ebs":
        MODEL_PATH = os.path.join("data", "model.pt")
    elif storage_backend == "nvme":
        MODEL_PATH = os.path.join("/opt/dlami/nvme/model.pt")
    elif storage_backend == "nfs":
        MODEL_PATH = os.path.join("/nfs/model.pt")
    else:
        raise ValueError(f"Unknown storage backend: {storage_backend}")

    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    for run in range(6):
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

        # Load the model to GPU
        if torch.cuda.is_available():
            start_time = time.time()
            model = model.to("cuda")
            torch.cuda.synchronize()
            to_gpu_time = time.time() - start_time

            start_time = time.time()
            model = model.to("cpu")
            torch.cuda.synchronize()
            from_gpu_time = time.time() - start_time
        else:
            to_gpu_time = -1
            from_gpu_time = -1

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        os.unlink(MODEL_PATH)

        # Log serialization and deserialization times
        if run > 0:
            with open(TIME_FILE_PATH, "a") as f:
                f.write(f"{instance},{model_name},{serialized_size},{storage_backend},{serialize_time:.4f},{deserialize_time:.4f},{to_gpu_time:.4f},{from_gpu_time:.4f}\n")

        time.sleep(1)


if __name__ == "__main__":
    init_data_dir()

    parser = argparse.ArgumentParser(description="Serialization Speed Benchmark")
    parser.add_argument("instance", default="g5.2xlarge", help="The AWS instance type to use for the benchmark.")
    args = parser.parse_args()

    print("Will test %d models." % len(MODELS_TO_TEST))
    for storage_backend in ["ebs", "nvme", "nfs"]:
        for model_name in MODELS_TO_TEST:
            benchmark_serialization_speed(model_name, storage_backend, args.instance)
