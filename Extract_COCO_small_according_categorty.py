import os
import json
import shutil

# ==== 配置路径 ====
# COCO 数据集路径
coco_root = r"D:\Jiao\dataset\COCO"  # 修改为你的 COCO 路径
image_dir = os.path.join(coco_root, "images/val2017")
annotation_file = os.path.join(coco_root, "annotations/instances_val2017.json")

# 输出文件夹
output_image_dir = os.path.join(coco_root, "Small_traffic_light/valimages")
output_annotation_file = os.path.join(coco_root, "Small_traffic_light/valannotations.json")

# 确保输出文件夹存在
os.makedirs(output_image_dir, exist_ok=True)

# ==== 目标类别映射 ====
target_categories = {
    "traffic light": 10,
   }

# ==== 读取 COCO 标注文件 ====
with open(annotation_file, "r") as f:
    coco_data = json.load(f)

# COCO 类别 ID 映射
coco_categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
selected_category_ids = set(target_categories.values())

# 过滤小目标
SMALL_OBJECT_AREA = 32 * 32  # 小目标面积阈值
selected_annotations = []
selected_image_ids = set()

for ann in coco_data["annotations"]:
    if ann["category_id"] in selected_category_ids and ann["area"] < SMALL_OBJECT_AREA:
        selected_annotations.append(ann)
        selected_image_ids.add(ann["image_id"])

# 过滤符合条件的图片信息
selected_images = [img for img in coco_data["images"] if img["id"] in selected_image_ids]

# 复制符合条件的图片
for img in selected_images:
    src_img_path = os.path.join(image_dir, img["file_name"])
    dest_img_path = os.path.join(output_image_dir, img["file_name"])

    if os.path.exists(src_img_path):
        shutil.copy(src_img_path, dest_img_path)

# 保存新的标注文件
filtered_coco_data = {
    "images": selected_images,
    "annotations": selected_annotations,
    "categories": coco_data["categories"],
}

with open(output_annotation_file, "w") as f:
    json.dump(filtered_coco_data, f, indent=4)

print(f"✅ 筛选完成，共筛选出 {len(selected_images)} 张图片，已保存到 {output_image_dir}")
print(f"✅ 生成的新标注文件已保存到 {output_annotation_file}")
