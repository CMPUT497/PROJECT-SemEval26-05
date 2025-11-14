import torch

# Load the weights
weights = torch.load('relevance_head.pt', map_location='cpu')

# Print parameter names and their corresponding shapes
for name, param in weights.items():
    print(f"Name: {name}, Shape: {param.shape}")
    print(f"Weights for {name}:\n{param}\n")
