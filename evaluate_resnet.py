import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ============================================================
# 1. Dataset paths
# ============================================================

dataset_path = r"C:\Users\User\Downloads\documents"

test_path = dataset_path + r"\test"


model_path = "resnet50_cifake_3epochs.pth"


# ============================================================
# 2. Image transformation
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================================
# 3. Load test dataset
# ============================================================

test_dataset = datasets.ImageFolder(
    test_path,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_dataset))
print("Classes:", test_dataset.classes)
print("Class mapping:", test_dataset.class_to_idx)


# ============================================================
# 4. Select GPU
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 5. Load ResNet-50
# ============================================================

print("\nLoading trained ResNet-50...")

model = models.resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# Load the CORRECTED model
model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model = model.to(device)

model.eval()

print("Model loaded successfully!")


# ============================================================
# 6. Evaluate
# ============================================================

all_predictions = []
all_labels = []

print("\nStarting evaluation...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

print("Evaluation completed!")


# ============================================================
# 7. Calculate metrics
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="binary",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="binary",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="binary",
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# 8. Display results
# ============================================================

print("\n========== RESULTS ==========")

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1-Score : {f1 * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("=============================")