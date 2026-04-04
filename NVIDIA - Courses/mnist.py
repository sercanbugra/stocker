import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision.transforms import ToTensor

# Visualization tools
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

train_set = torchvision.datasets.MNIST("./data/", train=True, download=True)
valid_set = torchvision.datasets.MNIST("./data/", train=False, download=True)

# Add a transform to convert images to tensors
train_set = torchvision.datasets.MNIST(
    "./data/", train=True, download=True, transform=ToTensor()
)
valid_set = torchvision.datasets.MNIST(
    "./data/", train=False, download=True, transform=ToTensor()
)

x_0, y_0 = train_set[0]
print(f"Image Shape: {x_0.shape}")  # Should show [1, 28, 28] if using ToTensor
print(f"Label: {y_0}")

print(f"Using device: {device}")
print(f"CUDA Available: {torch.cuda.is_available()}")


plt.imshow(x_0.squeeze(0), cmap="gray")  # Remove channel dimension for display
plt.title(f"Label: {y_0}")
plt.show()



