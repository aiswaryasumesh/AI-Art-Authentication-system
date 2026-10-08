from torchvision import datasets, transforms

# CHANGE THIS PATH to the location of your extracted CIFAKE folder
dataset_path = r"C:\Users\User\Downloads\documents"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root=dataset_path + r"\train",
    transform=transform
)

print("Dataset loaded successfully!")
print("Number of images:", len(dataset))
print("Classes:", dataset.classes)
print("Class mapping:", dataset.class_to_idx)

image, label = dataset[0]

print("First image shape:", image.shape)
print("First image label:", label)