import os
from collections import Counter
import matplotlib.pyplot as plt

# ====== 配置字体以支持中文 ======
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

# ====== 配置路径 ======
base_dir = 'F:\jack_dataset\MTSD\yolo'
labels_dir = os.path.join(base_dir, 'labels')
classes_file = os.path.join(labels_dir, 'classes.txt')

# ====== 读取类别名 ======
with open(classes_file, 'r', encoding='utf-8') as f:
    classes = [line.strip() for line in f.readlines()]

def count_labels(label_subdir):
    """统计指定子目录下每个类别的数量"""
    counts = Counter()
    label_path = os.path.join(labels_dir, label_subdir)
    for file in os.listdir(label_path):
        if file.endswith('.txt'):
            with open(os.path.join(label_path, file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        counts[class_id] += 1
    return counts

def print_counts(counts, dataset_name):
    """打印每个类别的统计"""
    print(f'\n==== {dataset_name} 数据集类别数量 ====')
    for class_id, count in sorted(counts.items()):
        class_name = classes[class_id] if class_id < len(classes) else f'未知类别 {class_id}'
        print(f'类别 {class_id} ({class_name}): {count} 个样本')

def plot_counts(counts, dataset_name):
    """绘制类别数量的柱状图"""
    labels_list = [classes[cid] if cid < len(classes) else f'ID {cid}' for cid in counts.keys()]
    plt.figure(figsize=(14, 6))
    plt.bar(labels_list, counts.values())
    plt.xticks(rotation=90)
    plt.title(f'{dataset_name} 数据集类别数量分布')
    plt.xlabel('类别')
    plt.ylabel('数量')
    plt.tight_layout()
    plt.show()

# ====== 统计训练集和测试集 ======
train_counts = count_labels('train')
test_counts = count_labels('test')

# ====== 打印统计 ======
print_counts(train_counts, '训练集')
print_counts(test_counts, '测试集')

# # ====== 绘制图表 ======
# plot_counts(train_counts, '训练集')
# plot_counts(test_counts, '测试集')
