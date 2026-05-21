import cv2
import numpy as np
# Python示例代码片段，使用OpenCV
def add_fog(image, depth_map, fog_intensity=0.5):
    # 根据深度图添加雾效果
    fogged_image = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            fog_factor = min(1.0, max(0.0, depth_map[i][j] * fog_intensity))
            fogged_image[i][j] = (1 - fog_factor) * image[i][j] + fog_factor * np.array([160, 160, 160])
    return fogged_image
