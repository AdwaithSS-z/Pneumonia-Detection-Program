import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
class PneumoniaDetector(nn.Module):
    def __init__(self, num_classes=1):
        super(PneumoniaDetector, self).__init__()
        self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)

# Instantiate and load model
model = PneumoniaDetector().to(device)
model.load_state_dict(torch.load("pneumonia_efnetv2s_cropped.pth", map_location=device))
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to generate Grad-CAM heatmap
def generate_gradcam(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Select last convolutional layer
    target_layer = model.model.features[-1]  

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward and backward pass
    output = model(image_tensor)
    model.zero_grad()
    output.backward()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Compute Grad-CAM
    gradients = gradients.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()

    weights = np.mean(gradients, axis=(2, 3))  
    cam = np.sum(weights[:, :, None, None] * activations, axis=1)[0]

    cam = np.maximum(cam, 0)  
    cam = cam / np.max(cam)   

    # Resize heatmap to match input image dimensions
    img_cv = cv2.imread(image_path)
    original_size = (img_cv.shape[1], img_cv.shape[0])  # (width, height)

    cam = cv2.resize(cam, original_size)  # Resize heatmap

    # Apply Grad-CAM heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

    # Save the resized heatmap
    heatmap_path = os.path.join(UPLOAD_FOLDER, "heatmap.jpg")
    cv2.imwrite(heatmap_path, overlay)

    return heatmap_path


# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Get model prediction
            image_tensor = transform(Image.open(filepath).convert("RGB")).unsqueeze(0).to(device)
            output = model(image_tensor)
            prediction = "Pneumonia Detected" if output.item() > 0.5 else "No Pneumonia"
            confidence = f"{output.item() * 100:.2f}%"  # Convert to percentage

            # Generate Grad-CAM heatmap
            heatmap_path = generate_gradcam(filepath)

            return render_template(
                "index.html",
                image_path=filepath,
                heatmap_path=heatmap_path,
                prediction=prediction,
                confidence=confidence
            )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)