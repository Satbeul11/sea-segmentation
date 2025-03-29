import os
import json
import cv2
import random
from tqdm import tqdm
from PIL import Image

# 경로 설정
BASE_DIR = r"C:\Users\LEEJINSE\Desktop\sim\춘계학술대회_논문준비\train\해상 객체 이미지"
IMAGE_ROOT = os.path.join(BASE_DIR, "Training")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_yolo")
IMG_SIZE = (640, 640)

# 결과 저장 경로
IMG_SAVE_DIR = os.path.join(OUTPUT_DIR, "images")
LBL_SAVE_DIR = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(IMG_SAVE_DIR, exist_ok=True)
os.makedirs(LBL_SAVE_DIR, exist_ok=True)

# 모든 [원천] 폴더 경로 탐색
source_folders = [
    os.path.join(IMAGE_ROOT, d)
    for d in os.listdir(IMAGE_ROOT)
    if d.startswith("[원천]") and os.path.isdir(os.path.join(IMAGE_ROOT, d))
]

# 이미지-라벨 쌍 목록
data_pairs = []

for source_folder in source_folders:
    label_folder = source_folder.replace("[원천]", "[라벨]")

    for root, _, files in os.walk(source_folder):
        for fname in files:
            if fname.lower().endswith(".jpg"):
                image_path = os.path.join(root, fname)
                rel_path = os.path.relpath(image_path, source_folder)
                label_path = os.path.join(label_folder, rel_path).replace(".jpg", ".json")

                if os.path.exists(label_path):
                    data_pairs.append((image_path, label_path))

# 섞고 나누기
random.shuffle(data_pairs)
num_train = int(len(data_pairs) * 0.9)
train_pairs = data_pairs[:num_train]
val_pairs = data_pairs[num_train:]

# polygon -> YOLOv8 format

def process_label(label_path, img_w, img_h):
    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    yolo_lines = []
    for shape in data.get("shapes", []):
        if shape.get("category_id") != 2:
            continue
        if shape.get("type") != "polygon":
            continue

        seg = shape.get("segmentation", [])
        if not seg:
            continue

        norm_pts = []
        for i in range(0, len(seg[0]), 2):
            x = seg[0][i] / img_w
            y = seg[0][i + 1] / img_h
            norm_pts.extend([x, y])

        if len(norm_pts) >= 6:  # 최소 3점 이상
            yolo_lines.append("0 " + " ".join(map(str, norm_pts)))

    return yolo_lines

# 이미지 저장 및 라벨 저장 함수
def save_yolo_data(pairs, split):
    for img_path, lbl_path in tqdm(pairs, desc=f"Processing {split}"):
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize(IMG_SIZE)
        img_name = os.path.basename(img_path)

        save_img_path = os.path.join(IMG_SAVE_DIR, split, img_name)
        save_lbl_path = os.path.join(LBL_SAVE_DIR, split, img_name.replace(".jpg", ".txt"))

        os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
        os.makedirs(os.path.dirname(save_lbl_path), exist_ok=True)

        img_resized.save(save_img_path)

        yolo_lines = process_label(lbl_path, 3840, 2160)
        if yolo_lines:
            with open(save_lbl_path, 'w') as f:
                f.write("\n".join(yolo_lines))

save_yolo_data(train_pairs, "train")
save_yolo_data(val_pairs, "val")

print("\n✅ 변환 완료! YOLOv8 학습 준비 완료.")
