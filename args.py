import argparse

def get_args():
    parser = argparse.ArgumentParser(description='AWS GPU Benchmarking Arguments')
    parser.add_argument('--model', type=str, default="resnet18", help='Name of the model to use', choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--test_all', action='store_true', help='Test all models')
    
    args = parser.parse_args()
    return args
