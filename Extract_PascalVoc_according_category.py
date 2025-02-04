import os
import shutil

# Pascal VOC 数据集根目录
voc_root = "D:/Jiao/dataset/Pascal/VOCdevkit/VOC2012"
image_dir = os.path.join(voc_root, "JPEGImages")  # 原始图片目录
anno_dir = os.path.join(voc_root, "Annotations")  # 原始标注目录

# 选择要使用的 txt 文件
txt_file = os.path.join(voc_root, "ImageSets", "Main", "cat_trainval.txt")  # train + val
# txt_file = os.path.join(voc_root, "ImageSets", "Main", "cat_train.txt")  # 仅 train
# txt_file = os.path.join(voc_root, "ImageSets", "Main", "cat_val.txt")  # 仅 val

# 目标存储路径
output_dir = "D:/Jiao/dataset/Pascal/CatOnly2012"
output_images = os.path.join(output_dir, "JPEGImages")
output_annos = os.path.join(output_dir, "Annotations")

# 确保目标目录存在
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_annos, exist_ok=True)

# 读取所有包含猫的图像
cat_images = set()
with open(txt_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2 and parts[1] == "1":  # 1 表示这个图像包含猫
            cat_images.add(parts[0])  # 记录图像 ID

# 复制包含猫的图片和标注文件
for img_id in cat_images:
    src_img = os.path.join(image_dir, f"{img_id}.jpg")
    src_anno = os.path.join(anno_dir, f"{img_id}.xml")

    dst_img = os.path.join(output_images, f"{img_id}.jpg")
    dst_anno = os.path.join(output_annos, f"{img_id}.xml")

    if os.path.exists(src_img):
        shutil.copy(src_img, dst_img)
    if os.path.exists(src_anno):
        shutil.copy(src_anno, dst_anno)

print(f"成功提取 {len(cat_images)} 张含猫的图片和标注文件！")
