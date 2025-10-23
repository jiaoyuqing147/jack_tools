import os, glob, shutil, random
import numpy as np
from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
#pip install iterative-stratification
# ===== 配置 =====
src_labels_dir = r"E:\DataSets\FullIJCNN2013\labels"     # 只从这里读取每个txt
dst_root       = r"E:\DataSets\GTSDB\yolo43" # 输出到这里：labels/train|val|test
K = 43                                               # 重映射后类别数（按需改）
train_ratio, val_ratio, test_ratio = 0.70, 0.15, 0.15
keep_empty = False                # 空txt是否参与（当负样本）
ensure_full_coverage = True       # 力求让 val/test 覆盖尽量多的类
max_trials = 200
base_seed = 42

# 目标子目录（仅 labels）
labels_train = os.path.join(dst_root, "labels", "train")
labels_val   = os.path.join(dst_root, "labels", "val")
labels_test  = os.path.join(dst_root, "labels", "test")
for d in [labels_train, labels_val, labels_test]:
    os.makedirs(d, exist_ok=True)

# ===== 读取 src_labels_dir 下所有 .txt 并构造多标签向量 =====
label_paths = sorted(glob.glob(os.path.join(src_labels_dir, "*.txt")))
stems, Y = [], []
dropped_empty = 0

for lp in label_paths:
    classes = set()
    with open(lp, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            try:
                cid = int(parts[0])
            except:
                continue
            if 0 <= cid < K:
                classes.add(cid)

    if not classes and not keep_empty:
        dropped_empty += 1
        continue

    y = np.zeros(K, dtype=int)
    for c in classes:
        y[c] = 1

    Y.append(y)
    stems.append(lp)  # 直接保存完整label路径，后面copy方便

Y = np.array(Y, dtype=int)
N = len(stems)
assert N > 0, "没有可用的 .txt 标签文件（可能全是空且 keep_empty=False）"

print(f"可用标签文件数: {N} | 丢弃空文件数: {dropped_empty}")

def mls_split(Y, idxs, ratio, seed):
    """在 idxs 子集上做多标签分层，切出占比 ratio 的部分（尽量接近）"""
    m = len(idxs)
    part_size = int(round(m * ratio))
    part_size = max(1, min(part_size, m - 1))
    n_splits  = max(2, int(round(1 / ratio)))
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    subX = np.zeros((m, 1))
    subY = Y[idxs]
    best, min_gap = None, 1e9
    for tr, pt in mskf.split(subX, subY):
        gap = abs(len(pt) - part_size)
        if gap < min_gap:
            min_gap = gap
            best = (tr, pt)
    tr, pt = best
    remain = [idxs[i] for i in tr]
    part   = [idxs[i] for i in pt]
    return remain, part

def coverage(Y, idxs):
    s = Y[idxs].sum(axis=0)
    return int((s > 0).sum()), s  # 覆盖了多少类 & 每类出现计数

def try_split_with_seed(Y, seed):
    all_idxs = list(range(N))
    random.Random(seed).shuffle(all_idxs)
    # 先切 test
    remain, test = mls_split(Y, all_idxs, test_ratio, seed)
    # 再从剩余切 val
    val_ratio_on_remain = val_ratio / (1.0 - test_ratio)
    train, val = mls_split(Y, remain, val_ratio_on_remain, seed + 1)
    return train, val, test

# ===== 可选：搜索更好的 val/test 覆盖 =====
if ensure_full_coverage:
    best_result, best_score = None, (-1, -1)
    for t in range(max_trials):
        seed = base_seed + t
        tr, va, te = try_split_with_seed(Y, seed)
        cov_val, _ = coverage(Y, va)
        cov_test, _ = coverage(Y, te)
        score = (cov_val, cov_test)
        if score > best_score:
            best_score, best_result = score, (tr, va, te, seed)
            if cov_val == K and cov_test == K:
                break
    train_idxs, val_idxs, test_idxs, used_seed = best_result
    print(f"[覆盖优化] seed={used_seed} | Val覆盖 {best_score[0]}/{K} | Test覆盖 {best_score[1]}/{K}")
else:
    train_idxs, val_idxs, test_idxs = try_split_with_seed(Y, base_seed)

cov_train, _ = coverage(Y, train_idxs)
cov_val,   _ = coverage(Y, val_idxs)
cov_test,  _ = coverage(Y, test_idxs)
print(f"Train: {len(train_idxs)} | 覆盖 {cov_train}/{K}")
print(f"Val  : {len(val_idxs)} | 覆盖 {cov_val}/{K}")
print(f"Test : {len(test_idxs)} | 覆盖 {cov_test}/{K}")

# ===== 复制标签到目标 train/val/test 目录（不处理图像、不写清单）=====
def copy_labels(idxs, dst_dir):
    for i in idxs:
        src = stems[i]
        dst = os.path.join(dst_dir, os.path.basename(src))
        shutil.copy2(src, dst)

copy_labels(train_idxs, labels_train)
copy_labels(val_idxs,   labels_val)
copy_labels(test_idxs,  labels_test)

print("✅ 划分完成（仅标签）：")
print(f"Train labels: {labels_train}")
print(f"Val   labels: {labels_val}")
print(f"Test  labels: {labels_test}")
