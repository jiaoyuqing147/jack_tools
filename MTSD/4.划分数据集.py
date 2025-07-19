import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取 recognition 文件
recognition_file = 'GT_Recognition.txt'
recognition_df = pd.read_csv(recognition_file, sep=';', engine='python')

recognition_df['File Name'] = recognition_df['File Name'].str.strip("'").str.lower()
recognition_df['scene'] = recognition_df['File Name'].apply(lambda x: x.split('_')[0] + '.jpg')

scene_to_classes = recognition_df.groupby('scene')['Class ID'].apply(set).to_dict()

# 用 scene 中所有类别中频率最高的作为 major class
scene_major_class = {}
for scene, classes in scene_to_classes.items():
    class_counts = recognition_df[recognition_df['scene'] == scene]['Class ID'].value_counts()
    major = class_counts.idxmax()
    scene_major_class[scene] = major

scene_df = pd.DataFrame(list(scene_major_class.items()), columns=['scene', 'major_class'])

# 检查是否有类别数量不足
class_counts = scene_df['major_class'].value_counts()
if any(class_counts < 2):
    print('⚠ 存在只在1个 scene 中出现的 major_class，无法使用 stratify。将不使用 stratify，仅随机划分。')
    stratify = None
else:
    stratify = scene_df['major_class']

train_scenes, val_scenes = train_test_split(
    scene_df['scene'],
    test_size=0.3,
    random_state=42,
    stratify=stratify
)

print(f'训练集 scene 数: {len(train_scenes)}')
print(f'验证+测试集 scene 数: {len(val_scenes)}')

with open('train.txt', 'w') as f:
    for scene in train_scenes:
        f.write(f'{scene}\n')

with open('val.txt', 'w') as f:
    for scene in val_scenes:
        f.write(f'{scene}\n')

print('✅ 按 scene 分布划分完成，train.txt 和 val.txt 已保存')
