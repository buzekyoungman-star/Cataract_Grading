from pathlib import Path
import csv

import torch
from PIL import Image
from torchvision import models


# =========================================================
# 1. 项目路径
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "trial_100"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "predictions_imagenet_demo.csv"


# =========================================================
# 2. 运行设备
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备:", device)


# =========================================================
# 3. 加载 ImageNet 预训练 EfficientNet-B3
# =========================================================

weights = models.EfficientNet_B3_Weights.DEFAULT

model = models.efficientnet_b3(
    weights=weights
)

model = model.to(device)
model.eval()

print("EfficientNet-B3 加载成功")


# =========================================================
# 4. 图像预处理和 ImageNet 类别
# =========================================================

preprocess = weights.transforms()

categories = weights.meta["categories"]


# =========================================================
# 5. 搜索所有图片
# =========================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}

image_paths = sorted([
    path
    for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in image_extensions
])

print("找到图片数量:", len(image_paths))

if len(image_paths) == 0:
    raise RuntimeError(
        f"没有在 {IMAGE_DIR} 中找到图片"
    )


# =========================================================
# 6. 批量推理
# =========================================================

results = []

for number, image_path in enumerate(image_paths, start=1):

    try:
        image = Image.open(image_path).convert("RGB")

        input_tensor = preprocess(image)

        input_batch = (
            input_tensor
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

        confidence, predicted_index = torch.max(
            probabilities,
            dim=0
        )

        predicted_index = predicted_index.item()
        confidence = confidence.item()

        predicted_class = categories[predicted_index]

        results.append([
            image_path.name,
            predicted_index,
            predicted_class,
            confidence
        ])

        print(
            f"[{number}/{len(image_paths)}] "
            f"{image_path.name} -> "
            f"{predicted_class} "
            f"({confidence * 100:.2f}%)"
        )

    except Exception as e:
        print(
            f"[ERROR] {image_path.name}: {e}"
        )

        results.append([
            image_path.name,
            "",
            "ERROR",
            ""
        ])


# =========================================================
# 7. 保存 CSV
# =========================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "filename",
        "imagenet_class_index",
        "imagenet_prediction",
        "confidence"
    ])

    writer.writerows(results)


# =========================================================
# 8. 完成
# =========================================================

print()
print("=" * 60)

print(
    f"推理完成，共处理 {len(image_paths)} 张图片"
)

print(
    f"结果已保存到：{OUTPUT_CSV}"
)

print("=" * 60)