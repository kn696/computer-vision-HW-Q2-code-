import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 讀取影像
image_path = '1102-meng-of-Google4.jpg'  # 換成你的合照圖片檔名
img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb) # 注意 OpenCV 為 Y, Cr, Cb

# 2. 套用你觀察出的自訂門檻 (Cb: 110~135, Cr: 140~170)
# OpenCV 中的陣列順序為 [Y, Cr, Cb]
lower_skin = np.array([0, 140, 110], dtype=np.uint8)
upper_skin = np.array([255, 170, 135], dtype=np.uint8)

# 產生初步遮罩
mask_raw = cv2.inRange(img_ycbcr, lower_skin, upper_skin)

# 3. 形態學處理 (Morphological Operations) 對比證明
# 建立不同的結構元素 (Structuring Elements)
kernel_cross_3x3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
kernel_rect_5x5  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# (A) 3x3 十字形閉運算 (填補孔洞)
mask_cross_3x3 = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel_cross_3x3)

# (B) 5x5 矩形閉運算 (填補孔洞)
mask_rect_5x5 = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel_rect_5x5)

# (C) 最終優化：5x5 矩形閉運算後，再做開運算 (消除背景雜訊)
mask_final = cv2.morphologyEx(mask_rect_5x5, cv2.MORPH_OPEN, kernel_rect_5x5)

# 4. 提取最終膚色區域影像
result_skin = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_final)

# 5. 繪製報告所需的對比圖 (4 合 1)
plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title('1. Original Multi-ethnic Image', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(mask_raw, cmap='gray')
plt.title('2. Raw Skin Mask (Cb:110~135, Cr:140~170)\n[Contains interior holes & bg noise]', fontsize=11)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(mask_cross_3x3, cmap='gray')
plt.title('3. Morph Closing (3x3 Cross Kernel)\n[Incomplete filling of larger eye/mouth holes]', fontsize=11)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(mask_rect_5x5, cmap='gray')
plt.title('4. Morph Closing (5x5 Rect Kernel)\n[Effectively bridges and fills holes]', fontsize=11)
plt.axis('off')

plt.tight_layout()
plt.savefig('morphology_comparison_result.png', dpi=300)
plt.show()

# 6. 顯示最終膚色提取成果
plt.figure(figsize=(10, 6))
plt.imshow(result_skin)
plt.title('Final Segmented Skin Regions (Face, Hands, Arms)', fontsize=13, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('final_skin_result.png', dpi=300)
plt.show()