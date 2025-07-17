import pandas as pd
import random
import os
from collections import defaultdict

random.seed(42)

# ==== 加载 GT.csv ====
gt_df = pd.read_csv('GT.csv')
gt_df['filename'] = gt_df['filename'].str.strip().str.lower()

# 图片对应所有类别
image_to_classes = gt_df.groupby('filename')['Class ID'].apply(set)

# 类别 -> 包含该类别的图片集
class_to_images = defaultdict(set)
for img, classes in image_to_classes.items():
    for cls in classes:
        class_to_images[cls].add(img)

val_set = set()

# 步骤1：确保每类至少1张图片在验证集
for cls, imgs in class_to_images.items():
    available = list(imgs - val_set)
    if available:
        chosen = random.choice(available)
        val_set.add(chosen)

print(f"保证每类至少1张，初始验证集图片数：{len(val_set)}")

# 步骤2：剩余图片中，按7:3补足验证集
all_images = set(image_to_classes.index)
remaining_images = list(all_images - val_set)

target_val_size = int(len(all_images) * 0.3)
remaining_needed = target_val_size - len(val_set)

additional_val = random.sample(remaining_images, remaining_needed)
val_set.update(additional_val)

train_set = all_images - val_set

print(f"最终训练集: {len(train_set)} 张, 验证集: {len(val_set)} 张")

# ==== 统计类别分布 ====
def count_classes(filenames):
    subset = gt_df[gt_df['filename'].isin(filenames)]
    return subset['Class ID'].value_counts().sort_index()

train_class_counts = count_classes(train_set)
val_class_counts = count_classes(val_set)

with pd.option_context('display.max_rows', None):  # 显示所有行
    print("\n📊 训练集类别分布：\n", train_class_counts)
    print("\n📊 验证集类别分布：\n", val_class_counts)

train_class_counts.to_csv('splits/train_class_distribution.csv', header=['count'])
val_class_counts.to_csv('splits/val_class_distribution.csv', header=['count'])


# ==== 保存文件 ====
os.makedirs('splits', exist_ok=True)
pd.Series(list(train_set)).to_csv('splits/train_images.csv', index=False, header=False)
pd.Series(list(val_set)).to_csv('splits/val_images.csv', index=False, header=False)
