import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import cv2

# 📌 디렉터리 설정
BASE_DIR = r"C:\Users\LEEJINSE\Desktop\sim\춘계학술대회_논문준비\train\해상 객체 이미지"
IMAGE_BASE_DIR = os.path.join(BASE_DIR, "Validation")
MASK_BASE_DIR = os.path.join(BASE_DIR, "Val_mask_output")
MODEL_PATH = os.path.join(BASE_DIR, "결과", "unet_sea_segmentation_binary.h5")
IMG_SIZE = (512, 512)
NUM_CLASSES = 2

# 🔹 마스크 로드
def load_mask(mask_path):
    mask = np.load(mask_path)
    mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    mask = (mask == 255).astype(np.uint8)
    return to_categorical(mask, num_classes=NUM_CLASSES)

# 🔹 이미지 & 마스크 로드
def load_data(image_path, mask_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(IMG_SIZE)
        image = np.array(image) / 255.0
    except:
        return None, None

    if not os.path.exists(mask_path):
        return None, None

    mask = load_mask(mask_path)
    return image, mask

# 🔹 예측 시각화
def visualize_prediction(model, image_path, mask_path):
    image, true_mask = load_data(image_path, mask_path)
    if image is None or true_mask is None:
        print("❌ 로드 실패")
        return

    pred = model.predict(np.expand_dims(image, axis=0))[0]
    pred_class = np.argmax(pred, axis=-1)
    true_class = np.argmax(true_mask, axis=-1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("📷 Input Image")
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    plt.title("✅ Ground Truth")
    plt.imshow(true_class, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("🔮 Predicted Mask")
    plt.imshow(pred_class, cmap="gray")
    plt.show()

# 🔹 IoU 평가
def iou_score(y_true, y_pred):
    y_true = np.argmax(y_true, axis=-1).astype(bool)
    y_pred = np.argmax(y_pred, axis=-1).astype(bool)
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / union if union != 0 else 1.0

def evaluate_model_iou(model, image_paths, mask_paths, sample_size=30):
    ious = []
    for img_path, mask_path in zip(image_paths[:sample_size], mask_paths[:sample_size]):
        image, mask = load_data(img_path, mask_path)
        if image is None or mask is None:
            continue
        pred = model.predict(np.expand_dims(image, axis=0))[0]
        score = iou_score(mask, pred)
        ious.append(score)
    print(f"📊 평균 IoU (샘플 {len(ious)}개): {np.mean(ious):.4f}")



# 🔹 이미지-마스크 경로 매칭
image_paths = []
mask_paths = []

for root, _, files in os.walk(IMAGE_BASE_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg")):
            img_path = os.path.join(root, file)

            # 상대 경로 추출 (ex: 남해_여수항_8구역_SEG\20201205\0192\xxx.jpg)
            rel_path = os.path.relpath(img_path, IMAGE_BASE_DIR)
            path_parts = rel_path.split(os.sep)

            # 최상위 폴더명 변환: [원천] → [라벨]
            if path_parts[0].startswith("[원천]"):
                label_folder = "[라벨]" + path_parts[0][len("[원천]"):]
            else:
                print(f"❌ [원천] 폴더 아님: {img_path}")
                continue

            # 경로 조립
            rel_sub_path = os.path.join(*path_parts[1:])  # 하위 폴더 + 파일명
            mask_filename = os.path.splitext(os.path.basename(rel_sub_path))[0] + "_mask.npy"
            mask_subdir = os.path.dirname(rel_sub_path)
            mask_path = os.path.join(MASK_BASE_DIR, label_folder, mask_subdir, mask_filename)

            if os.path.exists(mask_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path)
            else:
                print(f"❌ 마스크 없음: {mask_path}")

print(f"✅ 평가 대상 이미지-마스크 쌍: {len(image_paths)}개")

# 🔹 모델 로드 & 평가 실행
model = tf.keras.models.load_model(MODEL_PATH)

# 다양한 샘플 수에 대해 평가
for sample_size in [25, 50, 95]:
    print(f"\n📦 샘플 개수: {sample_size}개")
    evaluate_model_iou(model, image_paths, mask_paths, sample_size=sample_size)

# 첫 이미지 시각화
visualize_prediction(model, image_paths[0], mask_paths[0])


# 🔹 방향키 시각화 + IoU
current_index = 0

def show_image(index):
    image, true_mask = load_data(image_paths[index], mask_paths[index])
    if image is None or true_mask is None:
        print("❌ 로드 실패")
        return

    pred = model.predict(np.expand_dims(image, axis=0))[0]
    pred_class = np.argmax(pred, axis=-1)
    true_class = np.argmax(true_mask, axis=-1)
    score = iou_score(true_mask, pred)

    plt.clf()
    filename = os.path.basename(image_paths[index])
    plt.suptitle(f"[{index+1}/{len(image_paths)}] {filename} | IoU: {score:.4f}", fontsize=12)

    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(true_class, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(pred_class, cmap="gray")

    plt.draw()
    print(f"📷 {filename} | IoU: {score:.4f}")

def on_key(event):
    global current_index
    if event.key == 'right':
        current_index = (current_index + 1) % len(image_paths)
    elif event.key == 'left':
        current_index = (current_index - 1) % len(image_paths)
    show_image(current_index)

fig = plt.figure(figsize=(12, 4))
fig.canvas.mpl_connect('key_press_event', on_key)
show_image(current_index)
plt.show()