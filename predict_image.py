import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# 1. Select device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. Load ResNet-50
# ============================================================

print("\nLoading ResNet-50...")

model = models.resnet50(weights=None)

# Replace final layer for 2 classes
model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# ============================================================
# 3. Load our trained 3-epoch model
# ============================================================

model.load_state_dict(
    torch.load(
        "resnet50_cifake_3epochs.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model loaded successfully!")


# ============================================================
# 4. Image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================================
# 5. Enter image path
# ============================================================

image_path = input(
    "\nEnter the full path of the image: "
)


# ============================================================
# 6. Load image
# ============================================================

try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    print("\nError loading image:", e)
    exit()


# ============================================================
# 7. Preprocess image
# ============================================================

image_tensor = transform(image)

# Add batch dimension
image_tensor = image_tensor.unsqueeze(0)

# Move to GPU
image_tensor = image_tensor.to(device)


# ============================================================
# 8. Make prediction
# ============================================================

with torch.no_grad():

    outputs = model(image_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()


# ============================================================
# 9. Get prediction probability
# ============================================================

confidence = probabilities[
    0,
    predicted_class
].item() * 100


# ============================================================
# 10. Class names
# ============================================================

class_names = [
    "FAKE",
    "REAL"
]

prediction = class_names[predicted_class]


# ============================================================
# 11. Display result
# ============================================================

print("\n====================================")
print("        AI ART AUTHENTICATION")
print("====================================")

print("Prediction :", prediction)
print(f"Confidence : {confidence:.2f}%")

print("====================================")