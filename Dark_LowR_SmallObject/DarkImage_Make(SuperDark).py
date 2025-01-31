import cv2
import os
import numpy as np
import shutil

# 设置输入和输出路径
image_dir = "F://jack_dataset//cocoalldata//Jack_generate_cat//COCO//train2017"  # 原始数据集路径
output_dir = r"F:\jack_dataset\cocoalldata\Jack_generate_cat\COCO\dark\train2017Dark"  # 低光照增强后的数据集路径
os.makedirs(output_dir, exist_ok=True)

# 亮度阈值
brightness_threshold = 50

def compute_brightness(image):
    """计算图像的亮度均值"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# 低光照增强方法
def gamma_correction(image, gamma=6.0):  # 调整 gamma
    """伽马变换 (gamma > 1.0 使图像变暗)"""
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def reduce_brightness(image, alpha=0.2, beta=0):  # 调整 alpha
    """降低亮度 (alpha < 1.0 使图像整体变暗)"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def add_gaussian_noise(image, mean=0, var=100):  # 调整 var
    """添加高斯噪声"""
    noise = np.random.normal(mean, var, image.shape).astype(np.uint8)
    return cv2.addWeighted(image, 0.6, noise, 0.4, 0)  # 增加噪声影响

# 统计处理的图像数量
count_gamma = 0
count_darker = 0
count_noisy = 0
count_original_low_light = 0

# 处理所有图片
for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"跳过无法读取的图像: {filename}")
            continue

        # 计算原始亮度
        original_brightness = compute_brightness(img)

        # 如果原始亮度 < 50，直接复制到目标文件夹
        if original_brightness < brightness_threshold:
            shutil.copy(img_path, os.path.join(output_dir, filename))
            count_original_low_light += 1
            continue  # 直接跳过增强步骤

        # 伽马变换
        gamma_dark = gamma_correction(img, gamma=6.0)
        gamma_brightness = compute_brightness(gamma_dark)
        print(f"{filename} - 伽马变换后亮度: {gamma_brightness}")

        if gamma_brightness < brightness_threshold:
            cv2.imwrite(os.path.join(output_dir, filename), gamma_dark)
            count_gamma += 1
        else:
            # 降低亮度
            darker = reduce_brightness(gamma_dark, alpha=0.2)
            darker_brightness = compute_brightness(darker)
            print(f"{filename} - 伽马+降低亮度后亮度: {darker_brightness}")

            if darker_brightness < brightness_threshold:
                cv2.imwrite(os.path.join(output_dir, filename), darker)
                count_darker += 1

        # 添加高斯噪声
        noisy = add_gaussian_noise(img, var=100)
        noisy_brightness = compute_brightness(noisy)
        print(f"{filename} - 添加噪声后亮度: {noisy_brightness}")

        if noisy_brightness < brightness_threshold:
            cv2.imwrite(os.path.join(output_dir, filename), noisy)
            count_noisy += 1

# 统计信息
print(f"伽马变换: {count_gamma}, 降低亮度: {count_darker}, 高斯噪声: {count_noisy}, 原始: {count_original_low_light}")
