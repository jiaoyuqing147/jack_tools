import cv2
import os

# ==== 配置路径 ====
# image_dir = r"E:\DataSets\MTSD\yolo54\images\train"
# label_dir = r"E:\DataSets\MTSD\yolo54\labels\train"
# output_dir = r"E:\DataSets\MTSD\yolo54\images\train_see"

# image_dir = r"E:\DataSets\tt100k_2021_paper2\tt100k_60_weather\images\test"
# label_dir = r"E:\DataSets\tt100k_2021_paper2\tt100k_60_weather\labels\test"
# output_dir = r"E:\DataSets\tt100k_2021_paper2\tt100k_60_weather\images\test_see"

image_dir = r"E:\DataSets\MTSD\yolo54_paper2\images\val"
label_dir = r"E:\DataSets\MTSD\yolo54_paper2\labels\val"
output_dir = r"E:\DataSets\MTSD\yolo54_paper2\images\val_see"

os.makedirs(output_dir, exist_ok=True)

# 支持的图片扩展名
image_extensions = ['.jpg', '.png', '.JPG', '.ppm', '.bmp', '.jpeg', '.JPEG', '.PNG', '.tif', '.tiff', '.TIFF', '.BMP',]

# 遍历标注文件
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        basename = os.path.splitext(filename)[0]

        # 尝试多种扩展名
        image_path = None
        for ext in image_extensions:
            candidate_path = os.path.join(image_dir, basename + ext)
            if os.path.exists(candidate_path):
                image_path = candidate_path
                break

        if image_path is None:
            print(f"❌ 跳过 {filename}，找不到对应的图像（支持扩展: {image_extensions}）")
            continue

        image = cv2.imread(image_path)
        height, width, _ = image.shape

        with open(os.path.join(label_dir, filename), "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                bbox_width = float(parts[3]) * width
                bbox_height = float(parts[4]) * height

                x1 = int(x_center - bbox_width / 2)
                y1 = int(y_center - bbox_height / 2)
                x2 = int(x_center + bbox_width / 2)
                y2 = int(y_center + bbox_height / 2)

                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"ID: {class_id}"
                cv2.putText(image, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        output_path = os.path.join(output_dir, basename + '_vis.jpg')
        cv2.imwrite(output_path, image)

        print(f"✅ 已处理 {filename}，保存为 {output_path}")

print("🎉 所有图片可视化完成！🚀")
