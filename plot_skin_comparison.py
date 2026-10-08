import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. 讀取 CSV
df = pd.read_csv('skin_samples.csv')

# 2. 將 RGB 轉換為 YCbCr
rgb_pixels = df[['R', 'G', 'B']].values.astype(np.uint8)
rgb_reshaped = rgb_pixels.reshape(-1, 1, 3)

# OpenCV 轉 YCrCb (順序為 Y, Cr, Cb)
ycrcb_reshaped = cv2.cvtColor(rgb_reshaped, cv2.COLOR_RGB2YCrCb)
ycrcb_pixels = ycrcb_reshaped.reshape(-1, 3)

df['Y'] = ycrcb_pixels[:, 0]
df['Cr'] = ycrcb_pixels[:, 1]
df['Cb'] = ycrcb_pixels[:, 2]

# 定義各類別在繪圖時的顏色與標記
color_map = {
    'White Skin': {'color': 'wheat', 'edge': 'orange', 'marker': 'o'},
    'Yellow Skin': {'color': 'gold', 'edge': 'goldenrod', 'marker': 's'},
    'Black Skin': {'color': 'saddlebrown', 'edge': 'black', 'marker': '^'},
    'Background': {'color': 'gray', 'edge': 'black', 'marker': 'x'}
}

# 3. 繪製圖表 (RGB 3D vs YCbCr 2D)
fig = plt.figure(figsize=(16, 7))

# --- 左圖：RGB 3D 散佈圖 ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
for category, group in df.groupby('Category'):
    cfg = color_map.get(category, {'color': 'blue', 'edge': 'k', 'marker': 'o'})
    ax1.scatter(group['R'], group['G'], group['B'], 
                c=cfg['color'], edgecolors=cfg['edge'], 
                marker=cfg['marker'], s=60, label=category, alpha=0.8)

ax1.set_title('RGB Color Space (3D)\n[Black & White skins are far apart]', fontsize=12, fontweight='bold')
ax1.set_xlabel('Red (R)')
ax1.set_ylabel('Green (G)')
ax1.set_zlabel('Blue (B)')
ax1.legend()

# --- 右圖：YCbCr Cb-Cr 2D 平面散佈圖 ---
ax2 = fig.add_subplot(1, 2, 2)
for category, group in df.groupby('Category'):
    cfg = color_map.get(category, {'color': 'blue', 'edge': 'k', 'marker': 'o'})
    ax2.scatter(group['Cb'], group['Cr'], 
                c=cfg['color'], edgecolors=cfg['edge'], 
                marker=cfg['marker'], s=70, label=category, alpha=0.85)

# 畫出經驗膚色框矩形 (Cb: 77~127, Cr: 133~173)
ax2.axvspan(77, 127, color='green', alpha=0.15, label='Skin Detection Zone')
ax2.axhline(y=133, color='r', linestyle='--', alpha=0.4)
ax2.axhline(y=173, color='r', linestyle='--', alpha=0.4)

ax2.set_title('YCbCr Color Space (Cb-Cr 2D Plane)\n[All skin tones clustered together]', fontsize=12, fontweight='bold')
ax2.set_xlabel('Cb (Chrominance Blue) [77 - 127]', fontsize=11)
ax2.set_ylabel('Cr (Chrominance Red) [133 - 173]', fontsize=11)
ax2.set_xlim(50, 150)
ax2.set_ylim(100, 200)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.savefig('skin_color_space_proof.png', dpi=300)
plt.show()