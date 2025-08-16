from collections import Counter
import os
import yaml
import matplotlib.pyplot as plt

# 替换为你的标签路径
label_dir = r'D:\Jiao\dataset\TsingHua100K\tt100k_2021\yolo\labels\train'

counter = Counter()
for file in os.listdir(label_dir):
    with open(os.path.join(label_dir, file), 'r') as f:
        for line in f:
            cls_id = int(line.split()[0])
            counter[cls_id] += 1

labels, freqs = zip(*sorted(counter.items(), key=lambda x: x[1], reverse=True))
plt.figure(figsize=(12,4))
plt.bar(range(len(labels)), freqs)
plt.title("类别频率分布")
plt.xlabel("类别ID")
plt.ylabel("样本数")
plt.show()
