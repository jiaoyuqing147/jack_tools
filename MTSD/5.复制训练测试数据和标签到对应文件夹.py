import pandas as pd
import os
import shutil

# ==== 配置路径 ====
source_image_dir = r'D:\Jiao\dataset\MTSD\MTSD\Detection'
source_label_dir = r'D:\Jiao\dataset\MTSD\MTSD\labels_yolo66'

train_list_path = 'train.txt'
val_list_path = 'val.txt'

train_image_dest = r'D:\Jiao\dataset\MTSD\MTSD\yolo\images\train'
val_image_dest = r'D:\Jiao\dataset\MTSD\MTSD\yolo\images\test'

train_label_dest = r'D:\Jiao\dataset\MTSD\MTSD\yolo\labels\train'
val_label_dest = r'D:\Jiao\dataset\MTSD\MTSD\yolo\labels\test'

# ==== 创建目标目录 ====
for path in [train_image_dest, val_image_dest, train_label_dest, val_label_dest]:
    os.makedirs(path, exist_ok=True)

# ==== 读取列表 ====
train_images = pd.read_csv(train_list_path, header=None)[0].tolist()
val_images = pd.read_csv(val_list_path, header=None)[0].tolist()

def copy_data(file_list, img_dest, label_dest):
    for filename in file_list:
        # 复制图片
        src_img_path = os.path.join(source_image_dir, filename)
        dst_img_path = os.path.join(img_dest, filename)

        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"❌ 图片不存在: {src_img_path}")

        # 复制标签 (把扩展名替换成 .txt)
        label_filename = os.path.splitext(filename)[0] + '.txt'
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
print(f"✅ 已复制 {len(val_images)} 张验证图片及标签")
