import torch
import torch.nn as nn
import torch.optim as optim
from requests import delete
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import json
import os
from PIL import Image

from CarBodyDataset import CarBodyDataset

# resizing and normalizing data to fit the model
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# initializing the dataset
data_path = os.path.join('data', 'train')
dataset = CarBodyDataset(data_path, 'mapping.json', transform=data_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# downloading the model
model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
num_classes = len(dataset.categories)

model.fc = nn.Linear(num_features, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 3
running_loss = 0

for epoch in range(epochs):
    model.train()

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # refreshing the gradients
        optimizer.zero_grad()
        outputs = model(inputs)
        # caluclating loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1} / {epochs} | Loss: {running_loss / len(dataloader)}")

torch.save(model.state_dict(), 'resnet18.pth')

train_results = {
    "epochs": epochs,
    "finall_loss" : running_loss / len(dataloader),
    "categories" : dataset.categories
}

with open("results.json", "w") as outfile:
    json.dump(train_results, outfile, indent=4)

print(f"Training finished, model and results saved.")


