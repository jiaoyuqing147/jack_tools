import os
import json

# ==== 设置路径 ====
coco_root = r"D:\Jiao\dataset\COCO\Small_traffic_light"  # 你的 COCO 目录
annotation_file = os.path.join(coco_root, "annotations.json")
image_dir = os.path.join(coco_root, "train2017")
output_label_dir = os.path.join(coco_root, "labels//train2017")

# 创建 YOLO 标注文件夹
os.makedirs(output_label_dir, exist_ok=True)

# 读取 COCO 标注文件
with open(annotation_file, "r") as f:
    coco_data = json.load(f)

# 获取图片信息
image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

# COCO 类别 ID 映射（traffic light 类别为 10，在 YOLO 里映射为 0）
category_mapping = {10: 0}  # 只保留 "traffic light" 类别，并映射为 YOLO 的 class_id = 0

# 处理标注数据
for ann in coco_data["annotations"]:
    category_id = ann["category_id"]

    # 只处理 traffic light 类别
    if category_id not in category_mapping:
        continue

    # 获取图片文件名
    image_id = ann["image_id"]
    image_filename = image_id_to_filename[image_id]
    image_width = next(img["width"] for img in coco_data["images"] if img["id"] == image_id)
    image_height = next(img["height"] for img in coco_data["images"] if img["id"] == image_id)

    # 获取目标框坐标
    x, y, width, height = ann["bbox"]

    # 计算 YOLO 格式的坐标（归一化）
    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    width = width / image_width
    height = height / image_height

    # YOLO class_id
    yolo_class_id = category_mapping[category_id]

    # 生成 YOLO 格式的标签文件
    txt_filename = os.path.splitext(image_filename)[0] + ".txt"
    txt_path = os.path.join(output_label_dir, txt_filename)

    # 写入 .txt 文件
    with open(txt_path, "a") as f:
        f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print(f"✅ 标注转换完成！YOLO 格式文件已保存至 {output_label_dir}")
