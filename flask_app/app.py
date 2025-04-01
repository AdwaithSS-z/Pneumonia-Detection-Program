import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
import datetime

# Initialize Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "static/uploads"
DATA_FILE = "patients_data.csv"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained pneumonia detection model (PneumoniaDetector)
class PneumoniaDetector(nn.Module):
    def __init__(self, num_classes=1):
        super(PneumoniaDetector, self).__init__()
        self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)

# Instantiate and load pneumonia detection model
model = PneumoniaDetector().to(device)
model.load_state_dict(torch.load("pneumonia_efnetv2s_best.pth", map_location=device))
model.eval()

# Load the pre-trained Grad-CAM model (efnetv2_best.pth)
class GradCAMModel(nn.Module):
    def __init__(self, num_classes=1):
        super(GradCAMModel, self).__init__()
        # Load EfficientNet V2 model for Grad-CAM
        self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)  # Modify for 2 classes (Pneumonia / No Pneumonia)

    def forward(self, x):
        return self.model(x)

gradcam_model = GradCAMModel().to(device)
gradcam_model.load_state_dict(torch.load("pneumonia_efnetv2s_cropped.pth", map_location=device))
gradcam_model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to generate Grad-CAM heatmap using the efnetv2_best model
def generate_gradcam(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Use the Grad-CAM model to generate the heatmap
    target_layer = gradcam_model.model.features[-1]  # Last convolutional layer of EfficientNet V2

    activations, gradients = None, None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    output = gradcam_model(image_tensor)
    gradcam_model.zero_grad()
    output.backward()

    forward_handle.remove()
    backward_handle.remove()

    gradients = gradients.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()

    weights = np.mean(gradients, axis=(2, 3))
    cam = np.sum(weights[:, :, None, None] * activations, axis=1)[0]

    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    img_cv = cv2.imread(image_path)
    original_size = (img_cv.shape[1], img_cv.shape[0])

    cam = cv2.resize(cam, original_size)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

    heatmap_path = os.path.join(UPLOAD_FOLDER, "heatmap.jpg")
    cv2.imwrite(heatmap_path, overlay)

    return heatmap_path

# Store patient data in a CSV file
def save_patient_data(name, age, phone, prediction, confidence):
    data = {
        "Name": [name],
        "Age": [age],
        "Phone": [phone],
        "Prediction": [prediction],
        "Confidence": [confidence]
    }
    df = pd.DataFrame(data)
    
    if not os.path.exists(DATA_FILE):
        df.to_csv(DATA_FILE, index=False)
    else:
        df.to_csv(DATA_FILE, mode="a", header=False, index=False)

# Generate PDF report
def generate_pdf(name, age, phone, diagnosis, confidence, xray_path, heatmap_path):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Add title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, height - 100, "PneumoScan Diagnostic Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 120, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Add patient information
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 160, "Patient Information:")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 180, f"Name: {name}")
    c.drawString(100, height - 200, f"Age: {age}")
    c.drawString(100, height - 220, f"Phone: {phone}")
    
    # Add diagnosis
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 260, "Diagnosis:")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 280, f"Result: {diagnosis}")
    c.drawString(100, height - 300, f"Confidence: {confidence}")
    
    # Add images
    img1 = ImageReader(xray_path)
    img2 = ImageReader(heatmap_path)
    
    # Original X-ray
    c.drawImage(img1, 100, height - 500, width=200, height=200, preserveAspectRatio=True)
    c.drawString(100, height - 510, "Original X-ray")
    
    # Heatmap
    c.drawImage(img2, 350, height - 500, width=200, height=200, preserveAspectRatio=True)
    c.drawString(350, height - 510, "Heatmap Analysis")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, height - 540, "Remarks:")  # You can add remarks here as needed
    
    # Footer
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(100, 50, "This report is generated by PneumoScan. For medical use only.")
    
    c.showPage()
    c.save()
    
    buffer.seek(0)
    return buffer

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        name = request.form["name"]
        age = request.form["age"]
        phone = request.form["phone"]

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Use the PneumoniaDetector model for prediction
            image_tensor = transform(Image.open(filepath).convert("RGB")).unsqueeze(0).to(device)
            output = model(image_tensor)
            prediction = "Pneumonia Detected" if output.item() > 0.5 else "No Pneumonia"
            confidence = f"{output.item() * 100:.2f}%"

            # Use the Grad-CAM model for Grad-CAM heatmap generation
            heatmap_path = generate_gradcam(filepath)

            # Save patient data
            save_patient_data(name, age, phone, prediction, confidence)

            return render_template(
                "index.html",
                image_path=filepath,
                heatmap_path=heatmap_path,
                prediction=prediction,
                confidence=confidence,
                patient_name=name,
                patient_age=age,
                patient_phone=phone
            )
    return render_template("index.html")

@app.route("/download_report", methods=["GET"])
def download_report():
    name = request.args.get("name")
    age = request.args.get("age")
    phone = request.args.get("phone")
    diagnosis = request.args.get("diagnosis")
    confidence = request.args.get("confidence")
    xray_path = request.args.get("xray_path")
    heatmap_path = request.args.get("heatmap_path")
    
    pdf_buffer = generate_pdf(name, age, phone, diagnosis, confidence, xray_path, heatmap_path)
    
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=f"PneumoScan_Report_{name.replace(' ', '_')}.pdf",
        mimetype="application/pdf"
    )

if __name__ == "__main__":
    app.run(debug=True)
