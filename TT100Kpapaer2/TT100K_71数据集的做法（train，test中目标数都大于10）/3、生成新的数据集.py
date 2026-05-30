import json
import shutil
from pathlib import Path
from PIL import Image
from collections import Counter
from pathlib import Path

for root in [
    Path(r"E:\DataSets"),
    Path(r"D:\DataSets"),
    Path("/home/jiaoyuqing/datasets"),
]:
    if root.exists():
        DATA_ROOT = root
        break
else:
    raise RuntimeError("Cannot find DATA_ROOT")

SRC_ROOT = DATA_ROOT / "tt100k_2021"
DST_ROOT = DATA_ROOT / "tt100k_2021_paper2" / "tt100k_71"

ANN_FILE = SRC_ROOT / "annotations_all.json"

# 是否清空旧输出目录
OVERWRITE = True

# ==========================
# 71类保留类别
# ==========================
KEEP_CLASSES = [
    "i10","i2","i2r","i4","i4l","i5",
    "il100","il60","il80","il90","im","ip",
    "p1","p10","p11","p12","p13","p14","p18","p19",
    "p23","p25","p26","p27","p29","p3","p5","p6","p9",
    "pa13","pa14",
    "pb","pbm","pbp","pcl","pdd","pg",
    "ph4","ph4.5","ph5",
    "pl10","pl100","pl120","pl15","pl20","pl30",
    "pl40","pl5","pl50","pl60","pl70","pl80",
    "pm20","pm30","pm55","pmb",
    "pn","pne","pr40","pr60","ps",
    "w13","w22","w30","w32","w55","w57","w58","w59","w63","wc"
]

class_to_id = {name: i for i, name in enumerate(KEEP_CLASSES)}

# ==========================
# 输出目录
# ==========================
if DST_ROOT.exists():
    if OVERWRITE:
        shutil.rmtree(DST_ROOT)
    else:
        raise FileExistsError(
            f"{DST_ROOT} 已存在。若要覆盖，请设置 OVERWRITE = True"
        )

for split in ["train", "test"]:
    (DST_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (DST_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

# ==========================
# 读取标注
# ==========================
with open(ANN_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# 原始图片路径索引
src_img_dirs = {
    "train": SRC_ROOT / "train",
    "test": SRC_ROOT / "test"
}

src_imgs = {}
for split, img_dir in src_img_dirs.items():
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            src_imgs[p.stem] = (split, p)

# ==========================
# 工具函数
# ==========================
def get_bbox(obj):
    """
    兼容 TT100K 常见 bbox 格式：
    bbox: {"xmin":..., "ymin":..., "xmax":..., "ymax":...}
    """
    bbox = obj.get("bbox", None)
    if bbox is None:
        return None

    xmin = bbox.get("xmin")
    ymin = bbox.get("ymin")
    xmax = bbox.get("xmax")
    ymax = bbox.get("ymax")

    if None in [xmin, ymin, xmax, ymax]:
        return None

    return float(xmin), float(ymin), float(xmax), float(ymax)


def voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    xmin = max(0, min(xmin, img_w - 1))
    xmax = max(0, min(xmax, img_w - 1))
    ymin = max(0, min(ymin, img_h - 1))
    ymax = max(0, min(ymax, img_h - 1))

    bw = xmax - xmin
    bh = ymax - ymin

    if bw <= 1 or bh <= 1:
        return None

    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    w = bw / img_w
    h = bh / img_h

    return x_center, y_center, w, h


# ==========================
# 转换数据
# ==========================
img_counter = Counter()
box_counter = Counter()
class_counter = Counter()

for img_name, img_info in data["imgs"].items():

    stem = Path(img_name).stem

    if stem not in src_imgs:
        continue

    split, img_path = src_imgs[stem]

    objects = img_info.get("objects", [])

    yolo_lines = []

    with Image.open(img_path) as im:
        img_w, img_h = im.size

    for obj in objects:
        cls_name = obj.get("category")

        if cls_name not in class_to_id:
            continue

        bbox = get_bbox(obj)
        if bbox is None:
            continue

        yolo_box = voc_to_yolo(*bbox, img_w, img_h)
        if yolo_box is None:
            continue

        cls_id = class_to_id[cls_name]
        x, y, w, h = yolo_box

        yolo_lines.append(
            f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        )

        box_counter[split] += 1
        class_counter[cls_name] += 1

    # 如果该图片没有保留类别，跳过不复制
    if len(yolo_lines) == 0:
        continue

    dst_img = DST_ROOT / "images" / split / img_path.name
    dst_label = DST_ROOT / "labels" / split / f"{img_path.stem}.txt"

    shutil.copy2(img_path, dst_img)

    with open(dst_label, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))

    img_counter[split] += 1

# ==========================
# 生成 classes.txt
# ==========================
classes_txt = DST_ROOT / "classes.txt"

with open(classes_txt, "w", encoding="utf-8") as f:
    for cls in KEEP_CLASSES:
        f.write(cls + "\n")

print(f"Classes file saved: {classes_txt}")

# ==========================
# 生成 data.yaml
# ==========================
yaml_path = DST_ROOT / "tt100k_paper2.yaml"

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(f"path: {DST_ROOT.as_posix()}\n")
    f.write("train: images/train\n")
    f.write("val: images/test\n")
    f.write("test: images/test\n")
    f.write(f"nc: {len(KEEP_CLASSES)}\n")
    f.write("names:\n")
    for i, name in enumerate(KEEP_CLASSES):
        f.write(f"  {i}: {name}\n")

# ==========================
# 输出统计
# ==========================
print("=" * 80)
print("DONE")
print("=" * 80)

print(f"Output: {DST_ROOT}")
print(f"YAML  : {yaml_path}")

print("\nImages:")
print(f"Train images: {img_counter['train']}")
print(f"Test images : {img_counter['test']}")

print("\nBoxes:")
print(f"Train boxes: {box_counter['train']}")
print(f"Test boxes : {box_counter['test']}")

print("\nClasses:")
print(f"Num classes: {len(KEEP_CLASSES)}")

print("\nClass mapping:")
for i, name in enumerate(KEEP_CLASSES):
    print(f"{i:2d}: {name}")

print("\nTop class counts:")
for cls, num in class_counter.most_common(20):
    print(f"{cls:10s}: {num}")