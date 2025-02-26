import os
import cv2

# 输入和输出路径
input_dir = r"D:\Jiao\dataset\JIAOTONGDENG\COCO\images\test2017"  # 原始图像目录
output_dir = r"D:\Jiao\dataset\JIAOTONGDENG\COCO\images\test2017resize4"  # 缩小后图像目录

# 确保输出文件夹存在
os.makedirs(output_dir, exist_ok=True)

# 处理所有图片
for filename in os.listdir(input_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):  # 支持常见格式
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 读取图像
        image = cv2.imread(input_path)
        if image is None:
            print(f"无法读取文件: {input_path}")
            continue

        # 获取原始尺寸
        height, width = image.shape[:2]

        # 计算缩小后的尺寸（1/4 尺寸 = 宽高各缩小 1/2）
        new_width, new_height = width // 4, height // 4

        # 调整图像大小
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 保存缩小后的图像
        cv2.imwrite(output_path, resized_image)

print("✅ 所有图片已缩小至 1/8 尺寸！")
