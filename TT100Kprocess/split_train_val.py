import os
import random
import shutil

# === 配置路径 ===
images_dir = r'G:\Jack_datasets\TT100K\tt100k_2021\yolo\images\train'
labels_dir = r'G:\Jack_datasets\TT100K\tt100k_2021\yolo\labels\train'

# 新输出目录
train_images_out = r'G:\Jack_datasets\TT100K\tt100k_2021\yolonewsplit\images\train'
val_images_out = r'G:\Jack_datasets\TT100K\tt100k_2021\yolonewsplit\images\val'
train_labels_out = r'G:\Jack_datasets\TT100K\tt100k_2021\yolonewsplit\labels\train'
val_labels_out = r'G:\Jack_datasets\TT100K\tt100k_2021\yolonewsplit\labels\val'

# 创建目录
os.makedirs(train_images_out, exist_ok=True)
os.makedirs(val_images_out, exist_ok=True)
os.makedirs(train_labels_out, exist_ok=True)
os.makedirs(val_labels_out, exist_ok=True)

# 获取所有图像文件名（.jpg）
images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
images.sort()
random.seed(42)
random.shuffle(images)

# 抽取 10% 做验证集
val_count = int(len(images) * 0.10)
val_images = images[:val_count]
train_images = images[val_count:]

def move_files(image_list, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    for img_file in image_list:
        label_file = img_file.replace('.jpg', '.txt')
        # 复制图像
        shutil.copy(os.path.join(src_img_dir, img_file), os.path.join(dst_img_dir, img_file))
        # 复制标签
        shutil.copy(os.path.join(src_lbl_dir, label_file), os.path.join(dst_lbl_dir, label_file))

# 拷贝图像和标签
move_files(train_images, images_dir, labels_dir, train_images_out, train_labels_out)
move_files(val_images, images_dir, labels_dir, val_images_out, val_labels_out)

print(f'划分完成：训练集 {len(train_images)} 张，验证集 {len(val_images)} 张。')
