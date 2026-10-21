import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

MODEL_PATH = "resnet50_cifake_3epochs.pth"
FOLDER = Path("validation_data/REAL")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

methods = {
    "Current resize": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]),
    "Preserve aspect ratio": transforms.Compose([
        transforms.Resize(224),
        transforms.Pad((0, 0, 0, 0)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor()
    ])
}

for name in methods:
    print(f"\n--- {name} ---")
    for path in sorted(FOLDER.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        try:
            with Image.open(path) as im:
                image = im.convert("RGB")
            tensor = methods[name](image).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0]
            predicted = "FAKE" if probs[0] > probs[1] else "REAL"
            print(f"{path.name}: {predicted} | FAKE {probs[0].item()*100:.2f}% | REAL {probs[1].item()*100:.2f}%")
        except Exception as e:
            print(f"{path.name}: ERROR: {e}")
