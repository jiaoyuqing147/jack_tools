import os
import json
import shutil
from collections import defaultdict

# ========= 1. 路径根据你自己的情况修改 =========
BASE_DIR = r"E:\DataSets\DFG"
IMG_DIR = os.path.join(BASE_DIR, "JPEGImages")
ANN_DIR = os.path.join(BASE_DIR, "DFG-tsd-annot-json")
OUT_DIR = os.path.join(BASE_DIR, "DFG_YOLO")
# ============================================

os.makedirs(OUT_DIR, exist_ok=True)

def convert_split(split_name):
    """
    split_name: 'train' 或 'test'
    """
    json_path = os.path.join(ANN_DIR, f"{split_name}.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) 建目录
    img_out_dir = os.path.join(OUT_DIR, "images", "train" if split_name == "train" else "val")
    label_out_dir = os.path.join(OUT_DIR, "labels", "train" if split_name == "train" else "val")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    # 2) 映射：image_id -> {file_name, width, height}
    id2img = {im["id"]: im for im in data["images"]}

    # 3) 按 image_id 聚合标注
    anns_by_img = defaultdict(list)
    for ann in data["annotations"]:
        # 如果有 iscrowd 字段且为 1，可以选择跳过
        if ann.get("iscrowd", 0) == 1:
            continue
        anns_by_img[ann["image_id"]].append(ann)

    print(f"[{split_name}] images: {len(id2img)}, anns: {len(data['annotations'])}")

    # 4) 逐图复制图片 & 生成 YOLO 标签
    for img_id, im in id2img.items():
        file_name = im["file_name"]
        width = im["width"]
        height = im["height"]

        # 4.1 复制图片
        src_img = os.path.join(IMG_DIR, file_name)
        dst_img = os.path.join(img_out_dir, file_name)
        if not os.path.exists(src_img):
            print("WARNING: image not found:", src_img)
            continue
        shutil.copy2(src_img, dst_img)

        # 4.2 生成标签文件
        label_path = os.path.join(label_out_dir, file_name.replace(".jpg", ".txt"))
        lines = []
        for ann in anns_by_img.get(img_id, []):
            x, y, w, h = ann["bbox"]   # COCO: [x_min, y_min, width, height]

            # YOLO 归一化中心点坐标
            xc = (x + w / 2) / width
            yc = (y + h / 2) / height
            ww = w / width
            hh = h / height

            # COCO 的 category_id 一般从 1 开始，YOLO 要从 0 开始,这里的json标注正好是从0开始的
            cls_id = ann["category_id"]

            line = f"{cls_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}"
            lines.append(line)

        # 即使没有目标，也要写一个空文件，YOLO 能识别“空图像”
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print(f"[{split_name}] done. images -> {img_out_dir}, labels -> {label_out_dir}")


if __name__ == "__main__":
    convert_split("train")
    convert_split("test")   # 用作 val/test
    print("All done.")
