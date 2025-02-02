#原始数据集过于明亮，此脚本可降低图像亮度，仅保留亮度低于50的图像
'''
使用了数据增强方法
降低亮度（Brightness Reduction）
添加噪声（Gaussian Noise）
伽马变换（Gamma Correction）
'''
import cv2
import os
import numpy as np
import shutil

# 设置输入和输出路径
#image_dir = r"F:\jack_dataset\cocoalldata\Jack_generate_cat\COCO\train2017"  # 原始数据集路径
image_dir = r"D:\Jiao\dataset\Jack_generate_cat\COCO\val2017"  # 原始数据集路径
#output_dir = r"F:\jack_dataset\cocoalldata\Jack_generate_cat\COCO\dark\train2017Dark"  # 低光照增强后的数据集路径
output_dir = r"D:\Jiao\dataset\Jack_generate_cat\COCO\val2017Dark"  # 低光照处理后的数据集路径
os.makedirs(output_dir, exist_ok=True)

# 亮度阈值
brightness_threshold = 50


def compute_brightness(image):
    """计算图像的亮度均值"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


# 低光照增强方法
def reduce_brightness(image, alpha=0.4, beta=0):
    """降低亮度 (alpha < 1.0 使图像变暗)"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def add_gaussian_noise(image, mean=0, var=30):
    """添加高斯噪声"""
    noise = np.random.normal(mean, var, image.shape).astype(np.uint8)
    return cv2.addWeighted(image, 0.8, noise, 0.2, 0)


def gamma_correction(image, gamma=2.5):
    """伽马变换 (gamma > 1.0 使图像变暗)"""
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


# 统计处理的图像数量
count_darker = 0
count_noisy = 0
count_gamma = 0
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
            #shutil.copy(img_path, os.path.join(output_dir, "original_" + filename))
            shutil.copy(img_path, os.path.join(output_dir, filename))
            count_original_low_light += 1
            continue  # 不进行增强处理，直接跳过

        # 应用增强
        darker = reduce_brightness(img, alpha=0.4)
        noisy = add_gaussian_noise(img)
        gamma_dark = gamma_correction(img, gamma=2.5)

        # 计算增强后的亮度
        darker_brightness = compute_brightness(darker)
        noisy_brightness = compute_brightness(noisy)
        gamma_brightness = compute_brightness(gamma_dark)

        # 仅保存亮度 < 50 的图片，并更新统计计数
        if darker_brightness < brightness_threshold:
            #cv2.imwrite(os.path.join(output_dir, "darker_" + filename), darker)
            cv2.imwrite(os.path.join(output_dir, filename), darker)
            count_darker += 1
        if noisy_brightness < brightness_threshold:
            #cv2.imwrite(os.path.join(output_dir, "noisy_" + filename), noisy)
            cv2.imwrite(os.path.join(output_dir, filename), noisy)
            count_noisy += 1
        if gamma_brightness < brightness_threshold:
            #cv2.imwrite(os.path.join(output_dir, "gamma_" + filename), gamma_dark)
            cv2.imwrite(os.path.join(output_dir, filename), gamma_dark)
            count_gamma += 1

# 打印统计信息
print(f"所有低光照增强图像已生成，并存储到: {output_dir}")
print(f"降低亮度处理的图像数量: {count_darker}")
print(f"添加高斯噪声处理的图像数量: {count_noisy}")
print(f"伽马变换处理的图像数量: {count_gamma}")
print(f"原本低亮度 (<50) 的图像数量: {count_original_low_light}")