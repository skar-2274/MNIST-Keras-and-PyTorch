import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm

# Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Measure runtime
start_time = time.time()

# Preprocess and Load Data
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    transform=ToTensor(),
    download=True
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    transform=ToTensor(),
    download=True
)

# Prepare for training with DataLoader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define Model
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) # 32*28*28
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # 64*28*28
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2) # Halves output to 64*14*14

        self.fc1 = nn.Linear(64 * 14 * 14, 128) # 3D Tensor to 1D
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # Conv1 + ReLU
        x = F.relu(self.conv2(x)) # Conv2 + ReLU
        x = self.pool(x) # Max Pooling (Reduces Size)

        x = torch.flatten(x, start_dim=1) # Flatten from 4D to 2D for FC layers
        x = F.relu(self.fc1(x)) # First fully connected layer
        x = self.fc2(x) # Final layer (no activation because CrossEntropyLoss applies softmax)
        return x

# Check if model exists
model_path = "fashion_mnist_model.pth"

model = MNISTModel().to(device)

# Train Model
def train_model():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining the model...")

    num_epochs = 5
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

        epoch_loss = total_loss / len(train_dataloader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    model.eval()
    torch.save(model.state_dict(), model_path)
    print("Model saved.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Trained model reloaded.")

# Evaluate Model
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded.")
else:
    train_model()

class_labels = {
    0: "T-shirt/Top",
    1: "Trousers",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneakers",
    8: "Bag",
    9: "Ankle Boot"
}

# Predictions
def predictions():
    model.eval()
    predictions = []
    actuals = []
    data_iter = iter(test_dataloader)
    images, labels = next(data_iter)
    images = images.to(device)

    for i in range(10):
        image, label = images[i].unsqueeze(0), labels[i].item()
        with torch.no_grad():
            prediction = model(image)
            predicted_label = torch.argmax(prediction, dim=1).item()
        predictions.append(predicted_label)
        actuals.append(label)

    print(f"Prediction: {[class_labels[p] for p in predictions]}")
    print(f"Actual: {[class_labels[a] for a in actuals]}")

predictions()

# Measure Runtime
end_time = time.time()
total_time = end_time - start_time

print(f"\nRuntime: {total_time:.2f} seconds")
