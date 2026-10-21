import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "resnet50_cifake_3epochs.pth"

IMAGE_PATH = IMAGE_PATH = r"C:\Users\User\Downloads\WhatsApp Image 2026-10-03 at 3.15.25 PM.jpeg"
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# Load the trained model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model = model.to(device)
model.eval()

# Match the preprocessing used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load the real photograph
image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# Display both class probabilities
with torch.no_grad():
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)[0]

print("\nRaw model logits:", logits[0].cpu().tolist())
print(f"FAKE probability: {probabilities[0].item() * 100:.6f}%")
print(f"REAL probability: {probabilities[1].item() * 100:.6f}%")

print("\nNote: These are model probabilities,")
print("not proof that the classification is correct.")