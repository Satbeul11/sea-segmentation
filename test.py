import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 설정
MODEL_PATH = r"C:\Users\LEEJINSE\Desktop\sim\춘계학술대회_논문준비\결과\unet_sea_segmentation.h5"
NEW_IMAGE_PATH = r"./42_12.png"  # 여기에 테스트할 이미지 경로 넣기
IMG_SIZE = (256, 256)
NUM_CLASSES = 6

# 모델 로드
model = load_model(MODEL_PATH)

# 이미지 불러오기 및 전처리
image = Image.open(NEW_IMAGE_PATH).convert("RGB")
image = image.resize(IMG_SIZE)
image_np = np.array(image) / 255.0
input_tensor = np.expand_dims(image_np, axis=0)

# 예측
prediction = model.predict(input_tensor)[0]
pred_mask = np.argmax(prediction, axis=-1)

# 시각화
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title("New Input Image")

plt.subplot(1, 2, 2)
plt.imshow(pred_mask, cmap="jet")
plt.title("Predicted Mask")
plt.show()
