import os
import json
from PIL import Image

# === 配置路径 ===
json_path = r"/home/jiaoyuqing/AlgorithmCodes/datasets/TT100K/tt100k_2021/annotations_all.json"
image_root = r"/home/jiaoyuqing/AlgorithmCodes/datasets/TT100K/tt100k_2021/yolo/images"
label_root = r"/home/jiaoyuqing/AlgorithmCodes/datasets/TT100K/tt100k_2021/yolo/labels"

# 创建输出根目录
os.makedirs(label_root, exist_ok=True)

# 加载 JSON 文件
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

types = data["types"]  # 所有类别名称
type_to_id = {name: idx for idx, name in enumerate(types)}  # 类别映射为数字 ID

# 坐标转换函数（YOLO 格式）
def convert_bbox(bbox, img_w, img_h):
    xmin, xmax = bbox["xmin"], bbox["xmax"]
    ymin, ymax = bbox["ymin"], bbox["ymax"]
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height

# 遍历所有图像标注
for img_id, img_data in data["imgs"].items():
    rel_path = img_data["path"]               # 如 train/12345.jpg
    subset = rel_path.split("/")[0]           # train / test / other
    image_full_path = os.path.normpath(os.path.join(image_root, rel_path))
    label_name = os.path.splitext(os.path.basename(rel_path))[0] + ".txt"
    label_path = os.path.join(label_root, subset, label_name)

    os.makedirs(os.path.dirname(label_path), exist_ok=True)  # 自动创建 train/test/other

    if not os.path.exists(image_full_path):
        print(f"[跳过] 图像不存在：{image_full_path}")
        continue

    try:
        with Image.open(image_full_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        print(f"[错误] 无法读取图像尺寸：{image_full_path}，原因：{e}")
        continue

    with open(label_path, 'w') as f:
        for obj in img_data["objects"]:
            bbox = obj["bbox"]
            category = obj["category"]
            class_id = type_to_id[category]
            x_center, y_center, w, h = convert_bbox(bbox, img_w, img_h)
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

print("✅ 所有标注转换完成！")

# 生成 classes.txt 文件
classes_txt_path = os.path.join(label_root, "classes.txt")
with open(classes_txt_path, "w", encoding="utf-8") as f:
    for cls_name in types:
        f.write(cls_name + "\n")

print(f"✅ classes.txt 已生成，共 {len(types)} 个类别，路径：{classes_txt_path}")
