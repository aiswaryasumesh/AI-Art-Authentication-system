
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(
    r"C:\Users\User\Desktop\AI_ART_AUTHENTICATION"
    r"\art_dataset\Real_AI_SD_LD_Dataset"
)
CHECKPOINT = Path("resnet50_artbench.pth")
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

class ArtBinaryDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.transform = transform
        extensions = {".jpg", ".jpeg", ".png", ".webp"}

        for folder in sorted((Path(root) / split).iterdir()):
            if not folder.is_dir():
                continue

            label = 0 if folder.name.startswith("AI_") else 1

            for path in folder.iterdir():
                if path.is_file() and path.suffix.lower() in extensions:
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

test_dataset = ArtBinaryDataset(ROOT, "test", transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=True,
)

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for batch_number, (images, labels) in enumerate(test_loader, start=1):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().tolist()

        all_predictions.extend(predictions)
        all_labels.extend(labels.tolist())

        if batch_number % 100 == 0:
            print(f"Evaluated batches: {batch_number}/{len(test_loader)}")

print("\nTest images:", len(all_labels))
print("Class mapping: 0 = AI-GENERATED, 1 = REAL")
print(f"Accuracy:  {accuracy_score(all_labels, all_predictions) * 100:.2f}%")
print(f"Precision (REAL): {precision_score(all_labels, all_predictions, pos_label=1, zero_division=0) * 100:.2f}%")
print(f"Recall (REAL):    {recall_score(all_labels, all_predictions, pos_label=1, zero_division=0) * 100:.2f}%")
print(f"F1-score (REAL):  {f1_score(all_labels, all_predictions, pos_label=1, zero_division=0) * 100:.2f}%")

print("\nConfusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(all_labels, all_predictions, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1],
    target_names=["AI-GENERATED", "REAL"],
    zero_division=0,
))
