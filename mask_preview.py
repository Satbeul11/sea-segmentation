import numpy as np
import matplotlib.pyplot as plt

# 마스크 경로 (예시)
mask_path = r"C:\Users\LEEJINSE\Desktop\sim\춘계학술대회_논문준비\train\해상 객체 이미지\mask_output\[라벨]남해_여수항_5구역_SEG\남해_여수항_5구역_SEG\20201205\0106\여수항_맑음_20201205_0106_0035_mask.npy"

# 마스크 불러오기
mask = np.load(mask_path)

# 시각화
plt.imshow(mask, cmap='gray')
plt.title("Sea Mask")
plt.axis('off')
plt.show()
