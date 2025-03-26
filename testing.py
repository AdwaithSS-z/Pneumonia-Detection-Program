import torch
import torchvision
import os

print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("Current Working Directory:", os.getcwd())  # Print current directory
print("Train Dataset Exists:", os.path.exists("datasets/chest_xray/train"))
