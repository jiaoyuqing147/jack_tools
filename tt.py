
import os
import json

# 你的 COCO 目录
coco_root = r"D:\Jiao\dataset\COCO\Small_traffic_light"
image_dir = os.path.join(coco_root, "images")
annotation_file = os.path.join(coco_root, "annotations.json")

# 读取标注文件
with open(annotation_file, "r") as f:
    coco_data = json.load(f)

# 获取标注中的图像文件名
annotated_images = {img["file_name"] for img in coco_data["images"]}

# 获取实际存在的图像文件名
existing_images = set(os.listdir(image_dir))

# 计算缺失的图片
missing_images = annotated_images - existing_images

print(f"📌 预计的图片数: {len(annotated_images)}")
print(f"✅ 实际存在的图片数: {len(existing_images)}")
print(f"❌ 缺失的图片数: {len(missing_images)}")

if missing_images:
    print("❌ 以下图片缺失:")
    print(list(missing_images)[:10])  # 只显示前10张
