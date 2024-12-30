import torch

# Load the checkpoint with map_location set to 'cpu'
checkpoint = torch.load('tfasr_checkpoint/tfasr_099.pth', map_location=torch.device('cpu'))

# Print the type of object loaded
print(type(checkpoint))

# Inspect the contents of the checkpoint
print(checkpoint.keys())
