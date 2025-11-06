import pandas as pd
from collections import Counter

# === 配置路径 ===
# gt_path = r"/home/jiaoyuqing/AlgorithmCodes/datasets/TrainIJCNN2013/gt.txt"
gt_path = r'/home/jiaoyuqing/bigspace/workspaceJack/datasets/FullIJCNN2013/FullIJCNN2013/gt.txt'

# 加载 gt.txt
gt_data = pd.read_csv(gt_path, sep=';', header=None, names=['image', 'x1', 'y1', 'x2', 'y2', 'class_id'])

# 统计每个类别数量
class_counts = Counter(gt_data['class_id'])

# 显示统计
print("各类别出现次数：")
for class_id, count in sorted(class_counts.items()):
    print(f"类别 {class_id}：{count} 次")

# 如果需要存储成 CSV 文件
class_count_df = pd.DataFrame.from_dict(class_counts, orient='index', columns=['count']).sort_index()
class_count_df.index.name = 'class_id'
class_count_df.to_csv('class_count.csv')

print("统计已保存到 class_count.csv")
