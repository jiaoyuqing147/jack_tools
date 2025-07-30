import pandas as pd
import os
import shutil

# ==== 配置路径 ====
source_image_dir = r'/home/jiaoyuqing/AlgorithmCodes/datasets/TrainIJCNN2013'
source_label_dir = r'/home/jiaoyuqing/AlgorithmCodes/datasets/GTSDB/labels'

train_list_path = r'./split_lists_7_3/train.txt'
val_list_path = r'./split_lists_7_3/test.txt'

train_image_dest = r'/home/jiaoyuqing/AlgorithmCodes/datasets/GTSDB/yolo/images/train'
val_image_dest = r'/home/jiaoyuqing/AlgorithmCodes/datasets/GTSDB/yolo/images/test'

train_label_dest = r'/home/jiaoyuqing/AlgorithmCodes/datasets/GTSDB/yolo/labels/train'
val_label_dest = r'/home/jiaoyuqing/AlgorithmCodes/datasets/GTSDB/yolo/labels/test'

# 支持的图片扩展名
image_extensions = ['.ppm', '.jpg', '.png']

# ==== 创建目标目录 ====
for path in [train_image_dest, val_image_dest, train_label_dest, val_label_dest]:
    os.makedirs(path, exist_ok=True)

# ==== 读取列表，转为str防止int引起join报错 ====
train_images = pd.read_csv(train_list_path, header=None, dtype=str)[0].tolist()
val_images = pd.read_csv(val_list_path, header=None, dtype=str)[0].tolist()

def copy_data(file_list, img_dest, label_dest):
    for filename in file_list:
        # ---- 复制图片 ----
        img_found = False
        for ext in image_extensions:
            candidate_path = os.path.join(source_image_dir, filename + ext)
            if os.path.exists(candidate_path):
                dst_img_path = os.path.join(img_dest, filename + ext)
                shutil.copy2(candidate_path, dst_img_path)
                img_found = True
                break
        if not img_found:
            print(f"❌ 图片不存在: {filename}（尝试扩展: {image_extensions}）")

        # ---- 复制标签 ----
        label_filename = filename + '.txt'
        src_label_path = os.path.join(source_label_dir, label_filename)
        dst_label_path = os.path.join(label_dest, label_filename)

        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)
        else:
            print(f"⚠️ 标签不存在: {src_label_path}")

# ==== 执行复制 ====
copy_data(train_images, train_image_dest, train_label_dest)
print(f"✅ 已复制 {len(train_images)} 张训练图片及标签")

copy_data(val_images, val_image_dest, val_label_dest)
print(f"✅ 已复制 {len(val_images)} 张测试图片及标签")
