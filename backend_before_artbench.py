from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn as nn
from torchvision import models, transforms

from PIL import Image
import io


# ============================================================
# 1. Create FastAPI application
# ============================================================

app = FastAPI(
    title="AI Art Authentication API",
    description="ResNet-50 based AI-generated image detection API"
)


# ============================================================
# 2. Allow the frontend to communicate with the backend
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 3. Select device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 4. Load ResNet-50
# ============================================================

print("Loading ResNet-50...")

model = models.resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# ============================================================
# 5. Load trained model
# ============================================================

model.load_state_dict(
    torch.load(
        "resnet50_cifake_3epochs.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("ResNet-50 loaded successfully!")


# ============================================================
# 6. Image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================================
# 7. Class names
# ============================================================

class_names = [
    "AI-GENERATED",
    "REAL"
]


# ============================================================
# 8. Test endpoint
# ============================================================

@app.get("/")
def home():

    return {
        "message": "AI Art Authentication API is running",
        "model": "ResNet-50",
        "device": str(device)
    }


# ============================================================
# 9. Prediction endpoint
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")


    # Preprocess
    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)


    # Prediction
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


    # Get prediction
    prediction = class_names[predicted_class]

    confidence = (
        probabilities[
            0,
            predicted_class
        ].item() * 100
    )


    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "model": "ResNet-50"
    }