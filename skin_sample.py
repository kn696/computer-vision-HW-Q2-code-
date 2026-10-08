import cv2
import pandas as pd
import numpy as np
import os

IMAGE_PATH = 'b676b41f5259415e950e63e76bb69bbb.jpg'  # 換成你的合照圖片路徑
OUTPUT_CSV = 'skin_samples2.csv'

# 定義採樣類別與對應顏色 (BGR 格式)
CLASSES = {
    '1': {'name': 'White Skin', 'color': (255, 200, 0)},      # 藍黃色標記
    '2': {'name': 'Yellow Skin', 'color': (0, 255, 255)},     # 黃色標記
    '3': {'name': 'Black Skin', 'color': (255, 0, 255)},      # 洋紅色標記
    '4': {'name': 'Background', 'color': (0, 0, 255)}         # 紅色標記
}

current_class = '1'
samples = []
click_history = []

def mouse_callback(event, x, y, flags, param):
    global samples, click_history, img_clean, current_class

    if event == cv2.EVENT_LBUTTONDOWN:
        b, g, r = img_clean[y, x]
        cls_info = CLASSES[current_class]
        
        samples.append({
            'R': r, 'G': g, 'B': b, 
            'Category': cls_info['name'], 
            'X': x, 'Y': y
        })
        click_history.append({'pos': (x, y), 'color': cls_info['color']})
        print(f"[{cls_info['name']}] (X={x}, Y={y}) -> R:{r}, G:{g}, B:{b}")
        update_display()

def update_display():
    global img_display, img_clean
    img_display = img_clean.copy()

    # 繪製所有點位
    for item in click_history:
        cv2.circle(img_display, item['pos'], 3, item['color'], -1)

    # 顯示目前選擇的採樣類別與統計
    y_offset = 30
    for key, info in CLASSES.items():
        count = sum(1 for s in samples if s['Category'] == info['name'])
        prefix = "-> " if key == current_class else "   "
        text = f"{prefix}[{key}] {info['name']}: {count} points"
        cv2.putText(img_display, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, info['color'], 2)
        y_offset += 25

    cv2.putText(img_display, "Press '1~4' to Switch Class | 'Z' Undo | 'S' Save | 'Q' Quit", 
                (10, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Multi-Skin Sampler", img_display)

def main():
    global img_clean, img_display, current_class

    if not os.path.exists(IMAGE_PATH):
        print(f"錯誤：找不到圖片 '{IMAGE_PATH}'")
        return

    img_clean = cv2.imread(IMAGE_PATH)
    
    # 縮放過大圖片
    h, w = img_clean.shape[:2]
    if max(h, w) > 1000:
        scale = 1000.0 / max(h, w)
        img_clean = cv2.resize(img_clean, (int(w * scale), int(h * scale)))

    img_display = img_clean.copy()
    cv2.namedWindow("Multi-Skin Sampler")
    cv2.setMouseCallback("Multi-Skin Sampler", mouse_callback)

    update_display()

    while True:
        key = cv2.waitKey(1) & 0xFF

        # 按 1, 2, 3, 4 切換類別
        if chr(key) in CLASSES:
            current_class = chr(key)
            print(f"\n>>> 切換至採樣類別: {CLASSES[current_class]['name']}")
            update_display()

        elif key in [ord('z'), ord('Z')]:
            if samples:
                removed = samples.pop()
                click_history.pop()
                print(f"撤銷: {removed['Category']}")
                update_display()

        elif key in [ord('s'), ord('S')]:
            if samples:
                df = pd.DataFrame(samples)
                df.to_csv(OUTPUT_CSV, index=False)
                print(f"\n[成功] 已儲存 {len(df)} 筆數據至 '{OUTPUT_CSV}'")

        elif key in [ord('q'), ord('Q'), 27]:
            if samples:
                df = pd.DataFrame(samples)
                df.to_csv(OUTPUT_CSV, index=False)
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()