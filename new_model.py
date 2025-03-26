import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
from torch.utils.data import DataLoader
import cv2
import numpy as np
from tqdm import tqdm
import os

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to crop the chest region
def crop_chest(image):
    """Automatically detect and crop only the chest region from an X-ray."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)  # Thresholding
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Ensure reasonable aspect ratio (avoid cropping too much)
        if w > h * 1.2:  # If too wide, keep center
            x = max(0, x + (w - h) // 2)
            w = h
        
        return image[y:y+h, x:x+w]  # Crop to detected chest region

    return image  # Return original if no contours found

# Custom dataset loader to preprocess images
class ChestXRayDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = crop_chest(image)  # Crop chest area
        image = transform(image)  # Apply transformations
        return image, label

# Image transformations with stronger augmentations
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert from NumPy array
    transforms.Resize((256, 256)),  # Resize larger before cropping
    transforms.CenterCrop((224, 224)),  # Focus on chest
    transforms.RandomRotation(degrees=15),  # Increased rotation
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),  # Improved augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset with chest cropping
train_data = ChestXRayDataset(root="datasets/chest_xray/train", transform=transform)
val_data = ChestXRayDataset(root="datasets/chest_xray/test", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Define the model
class PneumoniaDetector(nn.Module):
    def __init__(self, num_classes=1):
        super(PneumoniaDetector, self).__init__()
        self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)

# Instantiate model
model = PneumoniaDetector().to(device)

# Loss and optimizer with class weights
weights = torch.tensor([1.5])  # Increase weight for pneumonia class
criterion = nn.BCEWithLogitsLoss(pos_weight=weights.to(device))  # Better for imbalanced data
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)  # Reduce LR every 3 epochs

# Training function with accuracy tracking
def train(model, train_loader, criterion, optimizer, scheduler, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct, total = 0, 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()  # Adjust learning rate

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# Evaluation function
def evaluate(model, val_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy

# Train the model
train(model, train_loader, criterion, optimizer, scheduler, num_epochs=15)
evaluate(model, val_loader)

# Save the trained model
torch.save(model.state_dict(), "pneumonia_efnetv2s_improved.pth")
