import pandas as pd
from sklearn.model_selection import train_test_split
import os

gt_df = pd.read_csv('GT_Detection.txt', delimiter=';', encoding='latin1')
gt_df['File Name'] = gt_df['File Name'].str.strip().str.strip("'").str.lower()

# 每张图片里哪个类别数量最多
def most_frequent_class(group):
    return group['Class ID'].value_counts().idxmax()

file_classes = gt_df.groupby('File Name').apply(most_frequent_class).reset_index(name='dominant_class')

train_files, val_files = train_test_split(
    file_classes['File Name'],
    test_size=0.3,
    random_state=42,
    stratify=file_classes['dominant_class']
)

os.makedirs('splits', exist_ok=True)
with open('splits/train.txt', 'w') as f:
    f.write('\n'.join(train_files))

with open('splits/val.txt', 'w') as f:
    f.write('\n'.join(val_files))

print(f"训练集：{len(train_files)} 张")
print(f"验证集：{len(val_files)} 张")

# 类别分布统计
def count_classes(files):
    subset = gt_df[gt_df['File Name'].isin(files)]
    return subset['Class ID'].value_counts().sort_index()

print("\n📊 训练集类别分布：\n", count_classes(train_files))
print("\n📊 验证集类别分布：\n", count_classes(val_files))
