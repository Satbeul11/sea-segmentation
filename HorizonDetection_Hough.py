import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical

# 📌 디렉터리 설정
BASE_DIR = r"C:\Users\LEEJINSE\Desktop\sim\춘계학술대회_논문준비\train\해상 객체 이미지"
IMAGE_BASE_DIR = os.path.join(BASE_DIR, "Validation")
MASK_BASE_DIR = os.path.join(BASE_DIR, "Val_mask_output")
IMG_SIZE = (512, 512)
NUM_CLASSES = 2

def find_mask_file(mask_root, filename_wo_ext):
    for root, _, files in os.walk(mask_root):
        for file in files:
            if file == filename_wo_ext + "_mask.npy":
                return os.path.join(root, file)
    return None


# 🔹 마스크 로드
def load_mask(mask_path):
    mask = np.load(mask_path)
    mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    return (mask == 255).astype(np.uint8)

# 🔹 이미지 로드
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

# 🔹 수평선 기반 바다 마스크 생성
def get_sea_mask_by_hough(gray_img):
    kernel = np.ones((7, 7), np.uint8)
    erosion = cv2.erode(gray_img, kernel, iterations=1)
    gaussian_3 = cv2.GaussianBlur(erosion, (9, 9), 10.0)
    unsharp_image = cv2.addWeighted(erosion, 1.5, gaussian_3, -0.5, 0, erosion)

    edges = cv2.Canny(unsharp_image, 100, 200)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)

    h, w = gray_img.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if lines is None:
        return mask  # 선이 검출되지 않은 경우 빈 마스크 반환

    # 수평선에 가까운 선 찾기
    best_line = None
    min_diff = float('inf')
    for line in lines:
        for rho, theta in line:
            diff = abs(theta - np.pi / 2)
            if diff < min_diff:
                min_diff = diff
                best_line = (rho, theta)

    rho, theta = best_line
    if theta == 0.0:
        x = int(rho)
        mask[:, x:] = 255
    else:
        a = -1 / np.tan(theta)
        b = rho / np.sin(theta)
        for y in range(h):
            x = int((y - b) / a)
            if 0 <= x < w:
                mask[y:, x:] = 255
                break
    return (mask == 255).astype(np.uint8)

# 🔹 IOU 계산
def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union != 0 else 0.0

# 🔹 전체 폴더 순회 및 IOU 출력
ious = []
for root, _, files in os.walk(IMAGE_BASE_DIR):
    for file in files:
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')): continue

        image_path = os.path.join(root, file)
        filename_wo_ext = os.path.splitext(file)[0]
        mask_path = find_mask_file(MASK_BASE_DIR, filename_wo_ext)

        if not os.path.exists(mask_path):
            print(f"[경고] GT 마스크 없음: {file}")
            continue

        try:
            gray = load_image(image_path)
            gt = load_mask(mask_path)
            pred = get_sea_mask_by_hough(gray)
            iou = compute_iou(pred, gt)
            ious.append(iou)
            print(f"{file}: IOU = {iou:.4f}")
        except Exception as e:
            print(f"[오류] {file}: {e}")

# 🔹 최종 결과
if ious:
    mean_iou = np.mean(ious)
    print(f"\n📊 전체 평균 IOU: {mean_iou:.4f}")
else:
    print("IOU 계산 실패: 유효한 이미지 없음")