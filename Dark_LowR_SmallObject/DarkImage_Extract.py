
#此脚本是为了提取出特定文件夹中的亮度低于50的图像，这种图像很少的，这个脚本暂时不用了

import cv2
import os
import numpy as np

# 设置你的数据集路径

image_dir = "F://jack_dataset//cocoalldata//Jack_generate_cat//COCO//train2017"  # 请替换成你的文件夹路径
output_dir = "F://jack_dataset//cocoalldata//Jack_generate_cat//COCO//low_light_images"  # 低光照图片的存储目录
os.makedirs(output_dir, exist_ok=True)

# 亮度阈值
brightness_threshold = 50

# 存储低光照图像路径
low_light_images = []

# 遍历所有图片
for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # 确保是图片文件
        img_path = os.path.join(image_dir, filename)

        # 读取图像并转换为灰度图
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"跳过无法读取的图像: {filename}")
            continue

        # 计算亮度均值
        brightness = np.mean(img)

        # 如果亮度小于阈值，则存储
        if brightness < brightness_threshold:
            low_light_images.append(img_path)
            # 也可以把这些图片复制到新文件夹
            cv2.imwrite(os.path.join(output_dir, filename), img)

print(f"找到 {len(low_light_images)} 张低光照图片")

# 保存低光照图片的文件名列表
with open("low_light_images.txt", "w") as f:
    for img in low_light_images:
        f.write(img + "\n")

print("低光照图像列表已保存到 low_light_images.txt")
