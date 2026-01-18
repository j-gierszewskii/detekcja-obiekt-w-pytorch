# =============================
# DETEKCJA OBIEKTÓW
# =============================

import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from google.colab import files

# =============================
# 1. URZĄDZENIE
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Używane urządzenie:", device)

# =============================
# 2. ID KLASY "CAR" W COCO
# =============================
CAR_CLASS_ID = 3  # car w zbiorze COCO

# =============================
# 3. MODELE DETEKCYJNE
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

for model in models_dict.values():
    model.to(device)
    model.eval()

# =============================
# 4. WCZYTANIE OBRAZU
# =============================
uploaded = files.upload()
image_name = list(uploaded.keys())[0]

image_original = Image.open(image_name).convert("RGB")
transform = transforms.ToTensor()
image_tensor = transform(image_original).to(device)

# =============================
# 5. DETEKCJA – TYLKO SAMOCHODY
# =============================
threshold = 0.5
plt.figure(figsize=(16, 10))

summary = []

for i, (name, model) in enumerate(models_dict.items(), 1):
    image = image_original.copy()
    draw = ImageDraw.Draw(image)

    with torch.no_grad():
        output = model([image_tensor])[0]

    boxes = output["boxes"]
    scores = output["scores"]
    labels = output["labels"]

    car_count = 0
    conf_sum = 0

    for box, score, label in zip(boxes, scores, labels):
        if label.item() == CAR_CLASS_ID and score.item() >= threshold:
            car_count += 1
            conf_sum += score.item()

            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), "car", fill="red")

    avg_conf = conf_sum / car_count if car_count > 0 else 0
    summary.append((name, car_count, avg_conf))

    plt.subplot(2, 3, i)
    plt.imshow(image)
    plt.title(
        f"{name}\n"
        f"Samochody: {car_count}, "
        f"Śr. pewność: {avg_conf:.2f}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()

# =============================
# 6. PODSUMOWANIE
# =============================
print("\n=== PODSUMOWANIE ===")
for name, count, conf in summary:
    print(
        f"{name} | "
        f"Liczba samochodów: {count} | "
        f"Śr. pewność: {conf:.2f}"
    )
