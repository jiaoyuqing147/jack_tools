import cv2
import os

# 设置路径
image_dir = r"D:\Jiao\dataset\Jack_generate_cat\lowQulityDarkRepartition\images\train2017"  # 处理后的图像文件夹
label_dir = r"D:\Jiao\dataset\Jack_generate_cat\lowQulityDarkRepartition\annotations\train2017"  # 处理后的标注文件夹
output_dir = r"D:\Jiao\dataset\Jack_generate_cat\lowQulityDarkRepartition\images\visualized_labels"  # 保存可视化结果

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历所有标注文件
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        label_path = os.path.join(label_dir, filename)  # 标注文件路径
        image_path = os.path.join(image_dir, filename.replace(".txt", ".jpg"))  # 对应的图像路径

        # 确保图像存在
        if not os.path.exists(image_path):
            print(f"❌ 跳过 {filename}，找不到对应的图像")
            continue

        # 读取图像
        image = cv2.imread(image_path)
        height, width, _ = image.shape  # 获取图像的原始宽高

        # 读取 YOLO 标注
        with open(label_path, "r") as f:
            lines = f.readlines()

        # 遍历所有标注
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])  # 类别 ID
                x_center = float(parts[1]) * width  # 反归一化 X 坐标
                y_center = float(parts[2]) * height  # 反归一化 Y 坐标
                bbox_width = float(parts[3]) * width  # 反归一化 宽度
                bbox_height = float(parts[4]) * height  # 反归一化 高度

                # 计算左上角和右下角坐标
                x1 = int(x_center - bbox_width / 2)
                y1 = int(y_center - bbox_height / 2)
                x2 = int(x_center + bbox_width / 2)
                y2 = int(y_center + bbox_height / 2)

                # 画框（蓝色，线条宽度 2）
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # 画类别 ID
                label = f"ID: {class_id}"
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 保存可视化结果
        output_path = os.path.join(output_dir, filename.replace(".txt", "_vis.jpg"))
        cv2.imwrite(output_path, image)

        print(f"✅ 已处理 {filename}，结果保存到 {output_path}")

print("🎉 所有图片可视化完成！🚀")
