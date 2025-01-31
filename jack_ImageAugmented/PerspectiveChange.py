#创建视角变化的数据集，OpenCV 透视变换
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原图
image_path = "F://000000000684.jpg"  # 请替换成你的图片路径
img = cv2.imread(image_path)

# 获取图像尺寸
h, w = img.shape[:2]

# 定义原始的四个角点
src_pts = np.float32([
    [0, 0],      # 左上角
    [w-1, 0],    # 右上角
    [0, h-1],    # 左下角
    [w-1, h-1]   # 右下角
])

# 目标变换后的四个角点
dst_pts = np.float32([
    [w * 0.1, h * 0.2],  # 左上角向右下移动
    [w * 0.9, h * 0.05], # 右上角向左上移动
    [w * 0.2, h * 0.8],  # 左下角向右移动
    [w, h]               # 右下角保持不变
])

# 计算透视变换矩阵
matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

# 进行透视变换
warped_img = cv2.warpPerspective(img, matrix, (w, h))

# 保存并展示结果
warped_path = "F://warped_output.jpg"
cv2.imwrite(warped_path, warped_img)

# 显示原图和变换后的图片
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(1,2,2)
plt.title("Warped Image (Perspective Transform)")
plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))

plt.show()
