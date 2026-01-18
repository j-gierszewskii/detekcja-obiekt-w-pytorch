# =============================
# BENCHMARK DETEKCJI OBIEKTÓW
# =============================

import os
import time
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# =============================
# 1. URZĄDZENIE
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Używane urządzenie:", device)

# =============================
# 2. DATASET YOLO
# =============================
class YOLODataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.images = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

# =============================
# 3. DATALOADER
# =============================
transform = transforms.ToTensor()

val_dataset = YOLODataset(
    "/content/drive/MyDrive/projekt/dataset/val/images",
    transform=transform
)


val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False
)

print("Liczba obrazów walidacyjnych:", len(val_dataset))

# =============================
# 4. MODELE DETEKCYJNE
# =============================
models_dict = {
    "Faster R-CNN ResNet50 FPN":
        torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True),

    "Faster R-CNN MobileNet":
        torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True),

    "RetinaNet":
        torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True),

    "SSD300":
        torchvision.models.detection.ssd300_vgg16(pretrained=True),

    "FCOS":
        torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)
}

# =============================
# 5. BENCHMARK
# =============================
results = []

with torch.no_grad():
    for name, model in models_dict.items():
        model.to(device)
        model.eval()

        total_boxes = 0
        total_confidence = 0
        start_time = time.time()

        for images in val_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)[0]

            scores = outputs["scores"]
            boxes = outputs["boxes"]

            for score in scores:
                if score.item() > 0.5:   # próg detekcji
                    total_boxes += 1
                    total_confidence += score.item()

        elapsed = time.time() - start_time
        avg_confidence = (
            total_confidence / total_boxes
            if total_boxes > 0 else 0
        )

        results.append({
            "model": name,
            "detections": total_boxes,
            "avg_confidence": avg_confidence,
            "time": elapsed
        })

        print(f"{name}")
        print(f"  Liczba detekcji: {total_boxes}")
        print(f"  Średnia pewność: {avg_confidence:.2f}")
        print(f"  Czas: {elapsed:.2f}s\n")

# =============================
# 6. PODSUMOWANIE
# =============================
print("\n=== PODSUMOWANIE BENCHMARKU ===")
for r in results:
    print(
        f"{r['model']} | "
        f"Detekcje: {r['detections']} | "
        f"Śr. pewność: {r['avg_confidence']:.2f} | "
        f"Czas: {r['time']:.2f}s"
    )
