"""
X-ray Lung Classification — Legitimate Solution
Uses EfficientNet-B0 with transfer learning, proper train/val split.
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

# ─── Config ───
DATA_DIR = "./dataset/public/"
OUTPUT_PATH = "./working/submission.csv"
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
NUM_CLASSES = 3
LABEL_MAP = {"normal": 0, "pneumonia": 1, "covid": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./working", exist_ok=True)

# ─── Dataset ───
class XRayDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["image_id"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = LABEL_MAP[row["label"]]
        return image, label

# ─── Transforms ───
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─── Load data & split ───
labels_df = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))
train_df, val_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df["label"], random_state=42)

train_loader = DataLoader(XRayDataset(train_df, DATA_DIR, train_transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(XRayDataset(val_df, DATA_DIR, val_transform), batch_size=BATCH_SIZE, shuffle=False)

# ─── Model ───
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ─── Training loop ───
best_val_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = correct / total
    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {train_loss/len(train_loader):.4f} — Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "./working/best_model.pth")

# ─── Inference on test set ───
model.load_state_dict(torch.load("./working/best_model.pth"))
model.eval()

test_dir = os.path.join(DATA_DIR, "test")
test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(".png")])

predictions = []
for img_name in test_images:
    img_path = os.path.join(test_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    image = val_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(1).item()
    predictions.append({"image_id": img_name, "prediction": INV_LABEL_MAP[pred]})

# ─── Save submission ───
submission = pd.DataFrame(predictions)
submission.to_csv(OUTPUT_PATH, index=False)
print(f"Submission saved to {OUTPUT_PATH} ({len(submission)} rows)")
