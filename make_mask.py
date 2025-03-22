import os
import json
import numpy as np
from PIL import Image, ImageDraw

# 설정
root_train_path = r"C:\Users\LEEJINSE\Desktop\sim\춘계학술대회_논문준비\train\해상 객체 이미지"
input_root = os.path.join(root_train_path, "Training")
output_mask_root = os.path.join(root_train_path, "mask_output")
label_folder_prefix = "[라벨]"
target_label = "sea"
resize_size = (512, 512)

def create_sea_mask_from_json(json_path, image_size=(512, 512)):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'imageWidth' not in data or 'imageHeight' not in data:
        raise KeyError("'imageWidth' 또는 'imageHeight' 누락")

    width, height = data['imageWidth'], data['imageHeight']
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for shape in data['shapes']:
        if shape['label'] == target_label and shape['shape_type'] == 'polygon':
            try:
                points = [(float(x), float(y)) for x, y in shape['points']]
                draw.polygon(points, fill=255)
            except Exception as e:
                print(f"⚠️  Skip one shape in {json_path}: {e}")
                continue

    resized_mask = mask.resize(image_size, Image.NEAREST)
    return np.array(resized_mask)

def get_all_json_paths(root_dir):
    json_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json') and not file.endswith('_meta.json'):
                json_paths.append(os.path.join(root, file))
    return json_paths

def make_mask_dataset():
    label_dirs = [d for d in os.listdir(input_root) if d.startswith(label_folder_prefix)]
    for label_dir in label_dirs:
        label_path = os.path.join(input_root, label_dir)
        json_files = get_all_json_paths(label_path)
        for json_path in json_files:
            try:
                mask_array = create_sea_mask_from_json(json_path, resize_size)
                relative_path = os.path.relpath(json_path, label_path)
                mask_filename = os.path.splitext(relative_path)[0] + "_mask.npy"
                output_path = os.path.join(output_mask_root, label_dir, mask_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.save(output_path, mask_array)
                print(f"✔ Saved: {output_path}")
            except Exception as e:
                print(f"✘ Failed: {json_path}\n   Reason: {e}")

if __name__ == "__main__":
    make_mask_dataset()
