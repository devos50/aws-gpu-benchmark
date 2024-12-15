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

# Client function
def client(server_ip):
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

    start_time = time.time()

    # Send data
    socket.send(model_bytes)

    # Wait for acknowledgment
    socket.recv()

    end_time = time.time()
    print(f"Model sent. Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Transfer Benchmark with ZeroMQ")
    parser.add_argument("mode", choices=["server", "client"], help="Mode to run: server or client")
    parser.add_argument("--ip", type=str, help="IP address of the server (required for client mode)")
    args = parser.parse_args()

    if args.mode == "server":
        server()
    elif args.mode == "client":
        if not args.ip:
            raise ValueError("IP address is required for client mode")
        client(args.ip)
