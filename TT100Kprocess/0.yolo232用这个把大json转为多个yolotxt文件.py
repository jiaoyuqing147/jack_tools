import os
import json

# === 配置路径 ===
json_path = r"D:\Jiao\dataset\TsingHua100K\tt100k_2021\annotations_all.json"  # 标注 JSON 路径
label_root = r"D:\Jiao\dataset\TsingHua100K\tt100k_2021\labels_all"  # 输出标签目录
os.makedirs(label_root, exist_ok=True)

# 图像固定尺寸
img_w, img_h = 2048, 2048

# === 加载 JSON ===
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

types = data["types"]
type_to_id = {name: idx for idx, name in enumerate(types)}

# === 转换标注 ===
count = 0
for img_id, img_data in data["imgs"].items():
    rel_path = img_data["path"]  # 如 train/123.jpg
    subset = rel_path.split("/")[0]  # train/test/other
    label_name = os.path.splitext(os.path.basename(rel_path))[0] + ".txt"
    label_path = os.path.join(label_root, subset, label_name)

    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    with open(label_path, "w") as f:
        for obj in img_data["objects"]:
            category = obj["category"]
            if category not in type_to_id:
                continue
            class_id = type_to_id[category]
            bbox = obj["bbox"]
            xmin, xmax = bbox["xmin"], bbox["xmax"]
            ymin, ymax = bbox["ymin"], bbox["ymax"]
            # 归一化
            x_center = (xmin + xmax) / 2.0 / img_w
            y_center = (ymin + ymax) / 2.0 / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    count += 1

print(f"✅ 已完成标注转换，共生成 {count} 个 YOLO txt 文件。")

# === 输出 classes.txt
classes_txt_path = os.path.join(label_root, "classes.txt")
with open(classes_txt_path, "w", encoding="utf-8") as f:
    for cls_name in types:
        f.write(cls_name + "\n")
print(f"✅ classes.txt 已生成（{len(types)} 类），路径：{classes_txt_path}")
