import argparse

SUPPORTED_MODELS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "efficientnet-b7", "mobilenet_v3_large", "vit-base-patch16-224", "vit-large-patch16-224"]
SUPPORTED_DATASETS = ["cifar10", "cifar100", "tiny-imagenet"]

def get_args():
    parser = argparse.ArgumentParser(description='AWS GPU Benchmarking Arguments')
    parser.add_argument('--model', type=str, default="resnet18", help='Name of the model to use', choices=SUPPORTED_MODELS)
    parser.add_argument('--dataset', type=str, default="cifar10", help='Name of the dataset to use', choices=SUPPORTED_DATASETS)
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs for training')
    parser.add_argument('--test-all', action='store_true', help='Test all models')
    
    args = parser.parse_args()
    return args
