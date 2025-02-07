import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# YOLO 标签文件所在目录（修改为你的路径）
yolo_labels_dir = r"D:\Jiao\dataset\COCO\Small_traffic_light\labels\train2017"

# 统计数据
bbox_sizes = []  # 存储目标面积（相对图片面积的比例）
bbox_ratios = []  # 存储宽高比（w/h）

# 读取所有标注文件
def parse_yolo_labels(label_dir):
    """ 解析 YOLO 格式的标注文件 """
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(label_dir, filename), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, x_center, y_center, width, height = map(float, parts)
                        bbox_sizes.append(width * height)  # 计算面积
                        bbox_ratios.append(width / height)  # 计算宽高比

# 运行解析函数
parse_yolo_labels(yolo_labels_dir)

def plot_distribution(data, title, xlabel):
    """ 绘制数据分布图 """
    plt.figure(figsize=(8, 5))
    sns.histplot(data, bins=50, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.show()

# 绘制目标面积分布
plot_distribution(bbox_sizes, "Traffic Light BBox Size Distribution", "BBox Size (Relative to Image)")

# 绘制宽高比分布
plot_distribution(bbox_ratios, "Traffic Light Aspect Ratio Distribution", "Aspect Ratio (Width / Height)")

# 计算小目标比例
small_object_threshold = 0.05  # 5% 面积比例
small_object_ratio = sum(np.array(bbox_sizes) < small_object_threshold) / len(bbox_sizes)
print(f"小目标（面积 < 5%）占比: {small_object_ratio * 100:.2f}%")
