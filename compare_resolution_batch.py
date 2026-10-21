from pathlib import Path
import csv
import torch
from PIL import Image
from torchvision import models, transforms

ROOT = Path(r"C:\Users\User\Desktop\AI_ART_AUTHENTICATION")
IMAGE_DIR = ROOT / "validation_data" / "REAL"
CHECKPOINT = ROOT / "resnet50_cifake_3epochs.pth"
CSV_PATH = ROOT / "resolution_comparison.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
model.load_state_dict(state)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

extensions = {".jpg", ".jpeg", ".png", ".webp"}
images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in extensions
    and not p.stem.startswith("diagnostic_")
)

if not images:
    raise SystemExit("No genuine images found in " + str(IMAGE_DIR))

rows = []

def predict(image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu()
    index = int(probs.argmax())
    return (
        ["FAKE", "REAL"][index],
        round(float(probs[index]) * 100, 2),
        round(float(probs[0]) * 100, 2),
        round(float(probs[1]) * 100, 2)
    )

for path in images:
    try:
        with Image.open(path) as source:
            original = source.convert("RGB")
            original_size = f"{original.width}x{original.height}"

            original_result = predict(original)

            small = original.resize((32, 32))
            resized_result = predict(small)

        rows.append({
            "filename": path.name,
            "original_size": original_size,
            "original_prediction": original_result[0],
            "original_confidence_percent": original_result[1],
            "original_fake_probability_percent": original_result[2],
            "original_real_probability_percent": original_result[3],
            "downscaled_prediction": resized_result[0],
            "downscaled_confidence_percent": resized_result[1],
            "downscaled_fake_probability_percent": resized_result[2],
            "downscaled_real_probability_percent": resized_result[3],
            "prediction_changed": original_result[0] != resized_result[0]
        })

        print(
            f"{path.name}: "
            f"Original={original_result[0]} ({original_result[1]}%), "
            f"Downscaled={resized_result[0]} ({resized_result[1]}%)"
        )

    except Exception as exc:
        print(f"Skipped {path.name}: {exc}")

if not rows:
    raise SystemExit("No images could be processed.")

with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\n========== SUMMARY ==========")
print("Images tested:", len(rows))
print("Original predicted REAL:", sum(r["original_prediction"] == "REAL" for r in rows))
print("Downscaled predicted REAL:", sum(r["downscaled_prediction"] == "REAL" for r in rows))
print("Predictions changed:", sum(r["prediction_changed"] for r in rows))
print("CSV saved to:", CSV_PATH)
