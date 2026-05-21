# Python示例代码片段，使用OpenCV和NumPy库
import cv2
import numpy as np

def add_rain(image, rain_drops):
    # 合成雨滴效果
    for drop in rain_drops:
        x, y = drop['position']
        size = drop['size']
        angle = drop['angle']
        # 绘制雨滴
        cv2.line(image, (x, y), (int(x + size * np.cos(angle)), int(y + size * np.sin(angle))), (200, 200, 255), 1)
    return image
