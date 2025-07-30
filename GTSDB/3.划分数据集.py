import pandas as pd
import os
from sklearn.model_selection import train_test_split
from collections import defaultdict

# ==== 配置参数 ====
gt_txt_path = r'/home/jiaoyuqing/AlgorithmCodes/datasets/TrainIJCNN2013/gt.txt'
output_dir = './split_lists_7_3'
train_ratio = 0.7
test_ratio = 0.3
random_seed = 42

os.makedirs(output_dir, exist_ok=True)

# ==== 读取gt.txt标注 ====
gt_data = pd.read_csv(gt_txt_path, sep=';', header=None, names=['image', 'x1', 'y1', 'x2', 'y2', 'class_id'])

# 每张图片关联的所有类别
image_to_classes = defaultdict(set)
for _, row in gt_data.iterrows():
    img_name = row['image'].replace('.ppm', '')
    image_to_classes[img_name].add(row['class_id'])

image_list = []
class_list = []

for image, classes in image_to_classes.items():
    image_list.append(image)
    class_list.append(sorted(list(classes))[0])  # 取第一个类别用于分层依据

df = pd.DataFrame({'image': image_list, 'class_id': class_list})

# ==== 统计每个类别数量 ====
class_counts = df['class_id'].value_counts()
single_sample_classes = class_counts[class_counts == 1].index.tolist()

print(f"⚠️ 仅出现1次的类别有：{single_sample_classes}")

# ==== 过滤掉仅出现1次的类别 ====
df_filtered = df[~df['class_id'].isin(single_sample_classes)].reset_index(drop=True)

print(f"✅ 过滤后剩余样本数：{len(df_filtered)}")

# ==== 分层划分 训练集与测试集 ====
train, test = train_test_split(
    df_filtered,
    test_size=test_ratio,
    stratify=df_filtered['class_id'],
    random_state=random_seed
)

# ==== 保存划分列表 ====
def save_list(df_part, filename):
    with open(os.path.join(output_dir, filename), 'w') as f:
        for img in df_part['image']:
            f.write(img + '\n')

save_list(train, 'train.txt')
save_list(test, 'test.txt')

print(f"🎉 划分完成：训练集 {len(train)} 张，测试集 {len(test)} 张")
print(f"📂 文件已保存到：{output_dir}")
