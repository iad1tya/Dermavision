import torch
import torch.nn as nn
import torchvision.transforms as transforms
import json
import cv2
import numpy as np
from PIL import Image

# Load class labels
with open("models/class_labels.json", "r") as f:
    class_labels = json.load(f)

# Define CNN Model (Same as trained model)
class SkinCNN(nn.Module):
    def __init__(self, num_classes):
        super(SkinCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkinCNN(num_classes=len(class_labels)).to(device)
model.load_state_dict(torch.load("models/skin_model.pth", map_location=device))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def detect_face(image):
    """Detect and crop the face from the image"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        return image[y:y+h, x:x+w]
    return None

def analyze_skin(image):
    """Analyze skin condition using CNN"""
    face = detect_face(image)
    if face is None:
        return "Face not detected"

    image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        predicted_class = class_labels[torch.argmax(outputs).item()]

    return predicted_class
