
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

CHECKPOINT = Path("resnet50_artbench.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

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


def predict_image(image_path):
    image_path = Path(image_path)

    if not image_path.is_file():
        print("Image not found:", image_path)
        return

    with Image.open(image_path) as img:
        image = transform(img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(image), dim=1)[0]

    ai_probability = probabilities[0].item() * 100
    real_probability = probabilities[1].item() * 100

    prediction = (
        "AI-GENERATED"
        if ai_probability >= real_probability
        else "REAL"
    )

    print("\nPrediction:", prediction)
    print(f"AI-generated score: {ai_probability:.2f}%")
    print(f"Real-image score:   {real_probability:.2f}%")
    print("\nNote: Scores are model confidence, not proof of authenticity.")


if __name__ == "__main__":
    image_path = input("Enter the full image path: ").strip().strip('"')
    predict_image(image_path)
