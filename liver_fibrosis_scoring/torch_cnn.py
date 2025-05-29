# Complete torch_cnn1.py script for Liver Fibrosis Classification with Training & Evaluation Stats
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Path to image dataset
image_folder = "Fibrosis_Dataset/Dataset"
class_labels = ['F0', 'F1', 'F2', 'F3', 'F4']
image_paths, labels = [], []

# Load image paths and labels
for label in class_labels:
    class_folder = os.path.join(image_folder, label)
    for fname in os.listdir(class_folder):
        image_paths.append(os.path.join(class_folder, fname))
        labels.append(label)

data = pd.DataFrame({"Image_Path": image_paths, "Label": labels})

# Sample N images per class label (e.g., 40 per class)
#N = 40
#data = data.groupby("Label", group_keys=False).apply(lambda x: x.sample(min(len(x), N), random_state=42)).reset_index(drop=True)


train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['Label'], random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, stratify=train_data['Label'], random_state=42)

class_indices = {label: idx for idx, label in enumerate(sorted(class_labels))}

# Dataset definition
class LiverDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.loc[idx, 'Image_Path']
        label = class_indices[self.data.loc[idx, 'Label']]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Loaders
train_loader = DataLoader(LiverDataset(train_data, transform), batch_size=32, shuffle=True)
val_loader = DataLoader(LiverDataset(val_data, transform), batch_size=32, shuffle=False)
test_loader = DataLoader(LiverDataset(test_data, transform), batch_size=32, shuffle=False)

# Model setup
model = models.densenet121(pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(class_labels))
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Training loop
num_epochs = 50
best_val_loss = float('inf')
patience = 5
early_stop_counter = 0
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
    model.train()
    train_loss, correct_train, total_train = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)
    train_acc = correct_train / total_train
    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_acc)

    model.eval()
    val_loss, correct_val, total_val = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)
    val_acc = correct_val / total_val
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_acc)
    print(f"Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f}")

    scheduler.step(val_losses[-1])
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
        print("Model improved and saved.")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

# Save training logs
df_stats = pd.DataFrame({
    "epoch": list(range(1, len(train_losses)+1)),
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_acc": train_accuracies,
    "val_acc": val_accuracies,
    "Label": [l.item() if torch.is_tensor(l) else l for l in labels[:len(train_losses)]]
})
df_stats.to_json("training_stats.json", orient='records')

# Evaluation
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
cfm = confusion_matrix(y_true, y_pred)

with open("eval_report.json", "w") as f:
    json.dump({
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cfm.tolist(),
        "classes": class_labels
    }, f, indent=2)

print(f"Final Test Accuracy: {acc * 100:.2f}%")
