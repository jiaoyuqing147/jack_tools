import os
import json
from PIL import Image

# ========== 仅保留的 110 类别 ==========
my_names = [
    'pl80','p6','ph4.2','pa13','w58','pl90','p5','pm55','pl60','ip','p11','wc','i2r','w30','p23','pl15',
    'pm10','pss','w34','iz','pr70','pg','il80','pb','pbm','pm5','pm40','ph4','w45','i4','pl70','i14','pne',
    'pr60','p12','p3','pl5','w13','p14','i4l','pr30','p17','ph3','pl30','pr50','pcd','pbp','ps','pm8','w18',
    'p10','pn','pa14','p2','ph2.5','w55','i12','ph4.3','i10','i13','p26','w42','p13','pr40','p25','w41',
    'pl20','ph4.8','pl40','pmb','pr20','p18','pm50','i2','w22','w47','pl120','w32','pm15','ph5','pl10',
    'il60','w57','pl100','pr80','p16','pl110','w59','w20','ph2','p9','il100','ph2.4','p19','ph3.5','pcl',
    'pl35','p28','w3','pl25','p1','w46','w63','i5','il90','w21','pl50','ph2.2','pm2','i3'
]

# === 配置路径 ===
json_path  = r"E:/tt100k_2021/annotations_all.json"    # 标注
image_root = r"E:/tt100k_2021/yolo110/images"          # 图像根目录（下含 train/test/other）
label_root = r"E:/tt100k_2021/yolo110/labels"          # YOLO标签输出根目录（将生成 train/test/other）

os.makedirs(label_root, exist_ok=True)

# 加载 JSON
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 仅保留的类别 → 连续 id 映射（0..109）
type_to_id = {name: i for i, name in enumerate(my_names)}

def convert_and_clip(bbox, img_w, img_h):
    """将 {xmin,xmax,ymin,ymax} 转 YOLO 格式并裁剪到 [0,1]，返回 None 表示无效框"""
    xmin, xmax = float(bbox["xmin"]), float(bbox["xmax"])
    ymin, ymax = float(bbox["ymin"]), float(bbox["ymax"])
    # 修正可能的反向标注
    if xmax < xmin: xmin, xmax = xmax, xmin
    if ymax < ymin: ymin, ymax = ymax, ymin
    # 过滤零/负面积
    if xmax <= xmin or ymax <= ymin:
        return None
    # 归一化
    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    width    = (xmax - xmin) / img_w
    height   = (ymax - ymin) / img_h
    # 裁剪到 [0,1]
    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    width    = min(max(width,    0.0), 1.0)
    height   = min(max(height,   0.0), 1.0)
    # 过小框可选过滤（阈值可调）
    if width <= 0 or height <= 0:
        return None
    return x_center, y_center, width, height

num_imgs = 0
num_boxes_written = 0

for img_id, img_info in data.get("imgs", {}).items():
    rel_path = img_info["path"]  # 例如 "train/xxx.jpg" / "test/xxx.jpg" / "other/xxx.jpg"
    # 兼容不同分隔符
    parts = rel_path.replace("\\", "/").split("/")
    subset = parts[0]  # train / test / other
    image_full_path = os.path.normpath(os.path.join(image_root, *parts))
    label_dir = os.path.join(label_root, subset)
    os.makedirs(label_dir, exist_ok=True)
    label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(rel_path))[0] + ".txt")

    if not os.path.exists(image_full_path):
        print(f"[跳过] 图像不存在：{image_full_path}")
        continue

    try:
        with Image.open(image_full_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        print(f"[错误] 无法读取图像：{image_full_path}，原因：{e}")
        continue

    lines = []
    for obj in img_info.get("objects", []):
        category = obj.get("category")
        if category not in type_to_id:
            continue  # 不是 110 个内的类
        bbox = obj.get("bbox", {})
        conv = convert_and_clip(bbox, img_w, img_h)
        if conv is None:
            continue
        cid = type_to_id[category]
        x, y, w, h = conv
        lines.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    # 写入（允许空文件，以便 YOLO 正确识别为“无目标”图片）
    with open(label_path, "w", encoding="utf-8") as lf:
        if lines:
            lf.write("\n".join(lines) + "\n")

    num_imgs += 1
    num_boxes_written += len(lines)

print(f"✅ 转换完成：{num_imgs} 张图，写入 {num_boxes_written} 个框。")

# 写 classes.txt（按 110 类顺序）
classes_txt = os.path.join(label_root, "classes.txt")
with open(classes_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(my_names) + "\n")
print(f"✅ classes.txt 保存：{classes_txt}（{len(my_names)} 类）")
