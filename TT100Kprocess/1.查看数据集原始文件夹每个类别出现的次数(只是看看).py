import os
from collections import Counter

# ========== 修改这里 ==========
labels_dir = r"D:\Jiao\dataset\TsingHua100K\tt100k_2021\labels_all\train"
K = 232  # 总类别数
# =================================

counter = Counter()

# 遍历所有txt文件
for file in os.listdir(labels_dir):
    if not file.endswith(".txt"):
        continue
    file_path = os.path.join(labels_dir, file)
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(float(parts[0]))  # 确保解析正确
            if 0 <= cls_id < K:
                counter[cls_id] += 1

print("===== 统计结果 =====")
print(f"总类别数（固定）：{K}")
print(f"出现过的类别数：{len([c for c in counter if counter[c] > 0])}")
print(f"总框数：{sum(counter.values())}")

print("\n类别分布：")
for i in range(K):
    print(f"类 {i}: {counter.get(i, 0)}")
