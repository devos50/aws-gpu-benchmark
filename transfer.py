import os
import zmq
import time
import torch
from transformers import AutoModel
import argparse

# Configuration
MODEL_NAME = "bert-base-uncased"  # Replace with the model you want to use
SERVER_IP = "0.0.0.0"  # Use this for the server to listen on all interfaces
SERVER_PORT = 5555

# Server function
def server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{SERVER_IP}:{SERVER_PORT}")
    print(f"Server is listening on {SERVER_IP}:{SERVER_PORT}")

    while True:
        start_time = time.time()

        # Receive model data
        model_bytes = socket.recv()

        end_time = time.time()
        print(f"Received model. Time taken: {end_time - start_time:.2f} seconds")
        print(f"Model size: {len(model_bytes) / (1024 * 1024):.2f} MB")

        # Send acknowledgment
        socket.send(b"Model received")


def init_data_dir():
    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists("data/transfer_times.csv"):
        with open("data/transfer_times.csv", "w") as f:
            f.write("from_instance,from_az,to_instance,to_az,model_size,time\n")


def send_model(socket, model_bytes, model_size_mb):
    start_time = time.time()

    # Send data
    socket.send(model_bytes)

    # Wait for acknowledgment
    socket.recv()

    end_time = time.time()
    print(f"Model sent. Time taken: {end_time - start_time:.2f} seconds")

    # Log transfer time
    with open("data/transfer_times.csv", "a") as f:
        f.write(f"{args.from_instance},{args.from_az},{args.to_instance},{args.to_az},{model_size_mb:.2f},{end_time - start_time:.4f}\n")


def client(server_ip):
    init_data_dir()

    # Load the model
    print(f"Loading model {MODEL_NAME}...")
    model = AutoModel.from_pretrained(MODEL_NAME)

    # Serialize the model
    print("Serializing model...")
    model_data = torch.save(model.state_dict(), "model.pt", _use_new_zipfile_serialization=False)

    with open("model.pt", "rb") as f:
        model_bytes = f.read()

    model_size_mb = len(model_bytes) / (1024 * 1024)
    print(f"Model serialized. Size: {model_size_mb:.2f} MB")

    # Send the model
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{server_ip}:{SERVER_PORT}")
    print(f"Connected to server at {server_ip}:{SERVER_PORT}")

    for i in range(args.tries):
        print(f"Sending model {i + 1}/{args.tries}...")
        send_model(socket, model_bytes, model_size_mb)
        time.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Transfer Benchmark with ZeroMQ")
    parser.add_argument("mode", choices=["server", "client"], help="Mode to run: server or client")
    parser.add_argument("--ip", type=str, help="IP address of the server (required for client mode)")
    parser.add_argument("--tries", type=int, default=10, help="Number of times to send the model (default: 5)")

    parser.add_argument("--from-instance", type=str, help="Instance type of the sender (required for logging)")
    parser.add_argument("--to-instance", type=str, help="Instance type of the receiver (required for logging)")
    parser.add_argument("--from-az", type=str, help="Availability zone of the sender (required for logging)")
    parser.add_argument("--to-az", type=str, help="Availability zone of the receiver (required for logging)")

    args = parser.parse_args()

    if args.mode == "server":
        server()
    elif args.mode == "client":
        if not args.ip or not args.from_instance or not args.to_instance or not args.from_az or not args.to_az:
            raise ValueError("IP address and instance types are required for client mode")
        client(args.ip)
