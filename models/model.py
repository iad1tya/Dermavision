import torch
import torch.nn as nn
import torch.nn.functional as F

# Define CNN model architecture
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 🔹 **Fix: Adjust Fully Connected (FC) Layers to Match Saved Model**
        self.fc1 = nn.Linear(128 * 16 * 16, 128)  # ✅ Restored original size
        self.fc2 = nn.Linear(128, 4)  # ✅ Restored original size (4 classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten before passing to FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Load the trained model
def load_model(model_path="models/cnn_model.pth"):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load weights
    model.eval()  # Set to evaluation mode
    return model
