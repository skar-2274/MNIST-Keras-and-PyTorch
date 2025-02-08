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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Measure runtime
start_time = time.time()

# Preprocess and Load Data
training_data = datasets.MNIST(
    root="data",
    train=True,
    transform=ToTensor(),
    download=True
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    transform=ToTensor(),
    download=True
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define Model
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Check if model exists
model_path = "mnist_model.pth"

model = MNISTModel().to(device)

# Train Model
def train_model():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining the model...")

    num_epochs = 5
    for epoch in range(num_epochs):
        total_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}/5, Loss: {total_loss / len(train_dataloader):.4f}")

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

# Predictions
def predictions():
    model.eval()
    predictions = []
    actuals = []
    data_iter = iter(test_dataloader)
    images, labels = next(data_iter)
    images = images.to(device)

    for i in range(20):
        image, label = images[i].unsqueeze(0), labels[i].item()
        with torch.no_grad():
            prediction = model(image)
            predicted_label = torch.argmax(prediction, dim=1).item()
        predictions.append(predicted_label)
        actuals.append(label)

    print(f"Prediction: {predictions}")
    print(f"Actual: {actuals}")

predictions()

# Measure Runtime
end_time = time.time()
total_time = end_time - start_time

print(f"\nRuntime: {total_time:.2f} seconds")
