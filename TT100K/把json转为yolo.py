

import os
import json
from PIL import Image

# ========== 仅保留的 143 类别 ==========
my_names = ['pl80', 'p6', 'ph4.2', 'pa13', 'im', 'w58', 'pl90', 'il70', 'p5', 'pm55', 'pl60', 'ip', 'p11', 'pdd', 'wc', 'i2r', 'w30', 'pmr', 'p23', 'pl15', 'pm10', 'pss', 'p4', 'w34', 'pw3.5', 'iz', 'p1n', 'pr70', 'pg', 'il80', 'pb', 'pbm', 'pm5', 'pm40', 'ph4', 'w45', 'i4', 'pl70', 'i14', 'i11', 'p29', 'pne', 'pr60', 'ph4.5', 'p12', 'p3', 'pl5', 'w13', 'p14', 'i4l', 'pr30', 'p17', 'ph3', 'w35', 'pl30', 'pctl', 'pr50', 'pm35', 'i1', 'pcd', 'pbp', 'pcr', 'ps', 'pm8', 'w18', 'p10', 'pn', 'pa14', 'ph3.2', 'p2', 'ph2.5', 'w55', 'pw3', 'i12', 'ph4.3', 'phclr', 'i10', 'i13', 'p26', 'p8', 'w42', 'il50', 'p13', 'pr40', 'p25', 'w41', 'pl20', 'ph4.8', 'pnlc', 'ph2.1', 'pm30', 'pl40', 'pmb', 'pr20', 'p18', 'pm50', 'i2', 'w22', 'w47', 'pl120', 'w12', 'w32', 'pm15', 'ph5', 'pw3.2', 'pl10', 'il60', 'w57', 'pl100', 'pr80', 'p16', 'pl110', 'w59', 'w20', 'ph2', 'p9', 'il100', 'ph2.4', 'p19', 'ph3.5', 'pcl', 'pl35', 'p15', 'phcs', 'p28', 'w3', 'pl25', 'il110', 'p1', 'w46', 'pn-2', 'w63', 'pm20', 'pmblr', 'i5', 'il90', 'w21', 'p27', 'pl50', 'ph2.2', 'pm2', 'i3', 'pw4']


# === 配置路径 ===
json_path = r"F:\tt100k_2021\annotations_all.json"
image_root = r"F:\tt100k_2021\train"
label_root = r"F:\tt100k_2021\yolo_labels\train"

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
            category = obj["category"]
            if category not in my_names:
                continue  # 跳过不在 143 类别集合中的目标
            class_id = type_to_id[category]
            bbox = obj["bbox"]
            x_center, y_center, w, h = convert_bbox(bbox, img_w, img_h)
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

print("✅ 所有标注转换完成！")

# 生成 classes.txt 文件
classes_txt_path = os.path.join(label_root, "classes.txt")
with open(classes_txt_path, "w", encoding="utf-8") as f:
    for cls_name in types:
        if cls_name in my_names:
            f.write(cls_name + "\n")


print(f"✅ classes.txt 已生成，共 {len(types)} 个类别，路径：{classes_txt_path}")
