import os
import random

root = r"G:\Jack_datasets\Mapillary\Mapillary\mtsd_v2_fully_annotated"
split_occ_dir = os.path.join(root, "splits_occluded")

train_all = os.path.join(split_occ_dir, "train_occluded_all.txt")
val_all = os.path.join(split_occ_dir, "val_occluded_all.txt")

out_dir = os.path.join(root, "splits_final")
os.makedirs(out_dir, exist_ok=True)

random.seed(42)

# ======================
# 读取
# ======================
def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def save_txt(keys, path):
    with open(path, "w", encoding="utf-8") as f:
        for k in keys:
            f.write(k + "\n")

train_keys = read_txt(train_all)
val_keys_all = read_txt(val_all)

print("train_all:", len(train_keys))
print("val_all:", len(val_keys_all))

# ======================
# 抽 train
# ======================
train_sample = random.sample(train_keys, min(2000, len(train_keys)))

# ======================
# 抽 val + test（不重叠）
# ======================
random.shuffle(val_keys_all)

val_sample = val_keys_all[:500]
test_sample = val_keys_all[500:1000]

# ======================
# 保存
# ======================
save_txt(train_sample, os.path.join(out_dir, "train_2000.txt"))
save_txt(val_sample, os.path.join(out_dir, "val_500.txt"))
save_txt(test_sample, os.path.join(out_dir, "test_500.txt"))

print("\n已生成：")
print("train_2000.txt")
print("val_500.txt")
print("test_500.txt")