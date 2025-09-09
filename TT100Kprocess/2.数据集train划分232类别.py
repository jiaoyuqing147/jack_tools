# -*- coding: utf-8 -*-
import os
import glob
import shutil
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit#pip install iterative-stratification

# ====== 路径与参数（按需改）======
src_labels_dir = r"E:\Datasets\tt100k_2021\labels_all\train"  # 源标签（一堆txt）
dst_root       = r"E:\Datasets\tt100k_2021\yolojack\labels"   # 目标根目录
K              = 232     # 类别总数
val_ratio      = 0.20    # 验证集占比
random_seed    = 42
# =================================

dst_train_dir = os.path.join(dst_root, "train")
dst_val_dir   = os.path.join(dst_root, "val")
os.makedirs(dst_train_dir, exist_ok=True)
os.makedirs(dst_val_dir, exist_ok=True)

# 收集所有txt
label_files = sorted(glob.glob(os.path.join(src_labels_dir, "*.txt")))
if not label_files:
    raise FileNotFoundError(f"未找到任何标签文件：{src_labels_dir}\\*.txt")

# 构建多标签向量（按文件，文件里出现过的类标 1）
X_files = []
Y_multi = []
empty_files = 0

for lf in label_files:
    y = np.zeros(K, dtype=int)
    has_any = False
    with open(lf, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                c = int(float(parts[0]))
                if 0 <= c < K:
                    y[c] = 1
                    has_any = True
            except:
                pass
    if not has_any:
        empty_files += 1  # 记录空标注文件
    X_files.append(lf)
    Y_multi.append(y)

X_files = np.array(X_files)
Y_multi = np.stack(Y_multi, axis=0)

# 分层划分（基于多标签）
msss = MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=val_ratio, random_state=random_seed
)
train_idx, val_idx = next(msss.split(X_files, Y_multi))
train_files = X_files[train_idx]
val_files   = X_files[val_idx]

# 复制文件
def copy_many(files, dst_dir):
    n = 0
    for src in files:
        base = os.path.basename(src)
        dst  = os.path.join(dst_dir, base)
        shutil.copy2(src, dst)
        n += 1
    return n

n_train = copy_many(train_files, dst_train_dir)
n_val   = copy_many(val_files, dst_val_dir)

# 统计每边的框数（可选，方便检查）
def count_boxes(files):
    total = 0
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
    return total

train_boxes = count_boxes(train_files)
val_boxes   = count_boxes(val_files)

# 输出摘要
print("===== 划分完成 =====")
print(f"源标签文件总数：{len(label_files)}（空标注文件：{empty_files}）")
print(f"训练集：{n_train} 个文件，框数 {train_boxes}")
print(f"验证集：{n_val} 个文件，框数 {val_boxes}")
print(f"目标目录：\n - {dst_train_dir}\n - {dst_val_dir}")
