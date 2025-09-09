import os
import glob
import numpy as np
import pandas as pd

# ====== 配置路径 ======
train_labels_dir = r"E:\Datasets\tt100k_2021\yolojack\labels\train"
val_labels_dir   = r"E:\Datasets\tt100k_2021\yolojack\labels\val"
test_labels_dir  = r"E:\Datasets\tt100k_2021\yolojack\labels\test"

K = 232  # 总类别数

# ====== 统计函数 ======
def count_labels(label_dir, K):
    counts = np.zeros(K, dtype=int)
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    for lf in label_files:
        with open(lf, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    c = int(float(parts[0]))
                    if 0 <= c < K:
                        counts[c] += 1
                except:
                    pass
    return counts

# ====== 统计结果 ======
train_counts = count_labels(train_labels_dir, K)
val_counts   = count_labels(val_labels_dir, K)
test_counts  = count_labels(test_labels_dir, K)

# ====== 保存为DataFrame ======
df = pd.DataFrame({
    "class_id": np.arange(K),
    "train_count": train_counts,
    "val_count": val_counts,
    "test_count": test_counts
})

pd.set_option("display.max_rows", None)  # 显示所有行
pd.set_option("display.max_columns", None)  # 显示所有列

print(df)

# ====== 保存到CSV ======
# out_path = r"class_distribution_AfterDivided.csv"
# df.to_csv(out_path, index=False, encoding="utf-8-sig")
# print(f"统计结果已保存到 {out_path}")
