from pathlib import Path
import pandas as pd
from pathlib import Path

# ==========================
# 自动寻找数据集根目录
# ==========================
DATA_ROOTS = [
    Path(r"E:\DataSets\tt100k_2021_paper2"),
    Path(r"/home/jiaoyuqing/datasets/tt100k_2021_paper2"),
    Path(r"D:\DataSets\tt100k_2021_paper2"),
]

DATA_ROOT = next(
    (p for p in DATA_ROOTS if p.exists()),
    None
)

if DATA_ROOT is None:
    raise RuntimeError("Cannot find DATA_ROOT")

print(f"Using DATA_ROOT: {DATA_ROOT}")
# ==========================
# 修改成你的路径
# ==========================
LABEL_DIR = DATA_ROOT / "tt100k_60" / "labels" / "train"

# Linux示例
# LABEL_DIR = Path(
#     "/home/jiaoyuqing/datasets/tt100k_2021_paper2/tt100k_60/labels/train"
# )

# ==========================
# 统计
# ==========================
records = []

label_files = sorted(LABEL_DIR.glob("*.txt"))

for txt_file in label_files:

    with open(txt_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()
                 if line.strip()]

    num_boxes = len(lines)

    records.append({
        "image_name": txt_file.stem,
        "num_boxes": num_boxes
    })

df = pd.DataFrame(records)

# 按目标数排序
df = df.sort_values(
    by="num_boxes",
    ascending=True
).reset_index(drop=True)

# 保存
save_path = "train_box_statistics.csv"
df.to_csv(save_path, index=False)

# ==========================
# 输出统计信息
# ==========================
print("=" * 60)
print("统计完成")
print("=" * 60)

print(f"图片数量: {len(df)}")
print(f"总目标数: {df['num_boxes'].sum()}")

print()
print("目标数统计:")
print(df["num_boxes"].describe())

print()
print("前10张(目标最少):")
print(df.head(10))

print()
print("后10张(目标最多):")
print(df.tail(10))

print()
print(f"结果保存到: {save_path}")