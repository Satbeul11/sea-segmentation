import os
import json
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ✅ 카테고리 ID → 이름 매핑
CATEGORY_ID_TO_NAME = {
    1: "sky", 2: "sea", 3: "island", 4: "rock",
    5: "wharf", 6: "others", 7: "horizon", 8: "coastline"
}

# ✅ 카테고리 이름 → 색상 매핑 (BGR)
COLOR_MAP = {
    "sky": (255, 200, 100),
    "sea": (100, 100, 255),
    "island": (0, 200, 0),
    "rock": (128, 128, 128),
    "wharf": (100, 0, 200),
    "others": (200, 200, 0),
    "horizon": (0, 0, 255),
    "coastline": (0, 255, 255)
}

# 🔧 루트 디렉토리 (상대경로)
root_image_dir = r"./Training/[원천]남해_여수항_8구역_SEG/남해_여수항_8구역_SEG/20201207"
root_label_dir = r"./Training/[라벨]남해_여수항_8구역_SEG/남해_여수항_8구역_SEG/20201207"


# 📂 모든 이미지/라벨 경로 정리
image_label_pairs = []
for folder in sorted(os.listdir(root_image_dir)):
    image_subdir = os.path.join(root_image_dir, folder)
    label_subdir = os.path.join(root_label_dir, folder)
    if not os.path.isdir(image_subdir):
        continue

    for file in sorted(os.listdir(image_subdir)):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(image_subdir, file)
            json_path = os.path.join(label_subdir, os.path.splitext(file)[0] + ".json")
            image_label_pairs.append((img_path, json_path))

# ▶️ 이미지 표시 함수
def show_image(index):
    image_path, json_path = image_label_pairs[index]
    original = np.array(Image.open(image_path).convert("RGB"))
    labeled = original.copy()

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for shape in data["shapes"]:
            cat_id = shape.get("category_id", shape.get("label_id", 0))
            category = CATEGORY_ID_TO_NAME.get(cat_id, "unknown")
            points = np.array(shape["points"], dtype=np.int32)

            if category not in COLOR_MAP:
                continue

            if shape["type"] == "polygon":
                cv2.fillPoly(labeled, [points], COLOR_MAP[category])
            elif shape["type"] == "polyline":
                thickness = 5 if category == "horizon" else 2  # 🔹 여기에 추가!
                cv2.polylines(labeled, [points], isClosed=False, color=COLOR_MAP[category], thickness=thickness)

    # 🎨 범례 생성
    patches = []
    for label, color in COLOR_MAP.items():
        bgr = tuple(np.array(color) / 255.0)
        patch = mpatches.Patch(color=bgr, label=label)
        patches.append(patch)

    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize='medium')

    ax1.clear()
    ax2.clear()
    ax1.imshow(original)
    ax1.set_title(f"Original\n{os.path.basename(image_path)}")
    ax2.imshow(labeled)
    ax2.set_title(f"Labeled\n{os.path.basename(json_path)}")
    for ax in (ax1, ax2):
        ax.axis("off")
    fig.canvas.draw()

# 🧭 키 이벤트 함수
def on_key(event):
    global current_index  # ✅ 전역 변수로 선언
    if event.key == 'right':
        current_index = (current_index + 1) % len(image_label_pairs)
        show_image(current_index)
    elif event.key == 'left':
        current_index = (current_index - 1) % len(image_label_pairs)
        show_image(current_index)


# 📌 초기화 및 인터페이스 구성
current_index = 0
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
fig.canvas.mpl_connect('key_press_event', on_key)
show_image(current_index)
plt.show()
