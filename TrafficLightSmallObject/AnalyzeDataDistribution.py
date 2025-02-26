import os
import numpy as np
import matplotlib.pyplot as plt

# 设置路径
label_dir = r"D:\Jiao\dataset\S2TLD\S2TLD7201280\normal_2\labels"
image_width, image_height = 1280, 720  # 你的图像大小

# 存储目标框的宽高
widths = []
heights = []
ratios = []  # 宽高比
areas = []  # 目标面积

# 遍历所有 YOLO 标注文件
for label_file in os.listdir(label_dir):
    if label_file.endswith(".txt"):
        with open(os.path.join(label_dir, label_file), "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                _, x_center, y_center, w, h = map(float, parts)

                # 计算真实的像素尺寸
                real_w = w * image_width
                real_h = h * image_height

                widths.append(real_w)
                heights.append(real_h)
                ratios.append(real_w / real_h)
                areas.append(real_w * real_h)

# 画出目标尺寸分布直方图
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.hist(widths, bins=20, color="blue", alpha=0.7)
plt.xlabel("Bounding Box Width")
plt.ylabel("Count")
plt.title("目标框宽度分布")

plt.subplot(1, 3, 2)
plt.hist(heights, bins=20, color="green", alpha=0.7)
plt.xlabel("Bounding Box Height")
plt.ylabel("Count")
plt.title("目标框高度分布")

plt.subplot(1, 3, 3)
plt.hist(ratios, bins=20, color="red", alpha=0.7)
plt.xlabel("Width / Height Ratio")
plt.ylabel("Count")
plt.title("目标框宽高比分布")

plt.tight_layout()
plt.show()
