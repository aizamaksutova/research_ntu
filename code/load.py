import torch

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define tensor sizes
tensor_size = 10000000  # Adjust the size as per your requirements

# Create large tensors on CPU and transfer them to GPU
tensor1 = torch.randn(tensor_size).to(device)
tensor2 = torch.randn(tensor_size).to(device)

# Perform frequent data transfers between CPU and GPU
for i in range(1000):  # Adjust the number of iterations as per your requirements
    tensor1 = tensor1.to('cpu')
    tensor2 = tensor2.to(device)
    tensor1 = tensor1.to(device)
    tensor2 = tensor2.to('cpu')

print("Data transfers completed.")
