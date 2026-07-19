import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. 配置
# ============================================================

GT_CSV_PATH = Path("GT.csv")
OUTPUT_DIR = Path("distribution_analysis")

FILENAME_COLUMN = "filename"
CLASS_COLUMN = "Class ID"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 读取并检查 GT.csv
# ============================================================

if not GT_CSV_PATH.exists():
    raise FileNotFoundError(f"找不到文件：{GT_CSV_PATH.resolve()}")

gt_df = pd.read_csv(GT_CSV_PATH)

required_columns = {FILENAME_COLUMN, CLASS_COLUMN}
missing_columns = required_columns - set(gt_df.columns)

if missing_columns:
    raise ValueError(
        f"GT.csv 缺少必要字段：{sorted(missing_columns)}\n"
        f"当前字段为：{gt_df.columns.tolist()}"
    )

# 删除文件名或类别为空的记录
gt_df = gt_df.dropna(
    subset=[FILENAME_COLUMN, CLASS_COLUMN]
).copy()

# 统一文件名格式
gt_df[FILENAME_COLUMN] = (
    gt_df[FILENAME_COLUMN]
    .astype(str)
    .str.strip()
    .str.lower()
)

# 类别编号尽量转为整数
gt_df[CLASS_COLUMN] = pd.to_numeric(
    gt_df[CLASS_COLUMN],
    errors="raise"
).astype(int)


# ============================================================
# 3. 数据完整性检查
# ============================================================

duplicate_rows = gt_df.duplicated().sum()

if duplicate_rows > 0:
    print(f"⚠️ GT.csv 中存在 {duplicate_rows} 行完全重复记录。")
    print("当前脚本不会自动删除，因为同一类别可能存在多个不同目标框。")

num_images = gt_df[FILENAME_COLUMN].nunique()
num_classes = gt_df[CLASS_COLUMN].nunique()
num_instances = len(gt_df)

print("===== 数据集基本信息 =====")
print(f"图像总数：{num_images}")
print(f"类别总数：{num_classes}")
print(f"标注框总数：{num_instances}")

if num_classes != 66:
    print(f"⚠️ 当前实际检测到 {num_classes} 个类别，不是 66 个。")


# ============================================================
# 4. 每个类别的实例数量
# ============================================================

class_instance_counts = (
    gt_df[CLASS_COLUMN]
    .value_counts()
    .sort_index()
)

# 每个类别出现的图像数量
class_image_counts = (
    gt_df[
        [FILENAME_COLUMN, CLASS_COLUMN]
    ]
    .drop_duplicates()
    [CLASS_COLUMN]
    .value_counts()
    .sort_index()
)

# 建立完整类别编号
# 如果类别编号是 1 到 66，使用下面这一行
all_classes = list(range(1, 67))

# 如果你的类别编号实际是 0 到 65，请改成：
# all_classes = list(range(66))

class_statistics_df = pd.DataFrame(
    index=all_classes
)

class_statistics_df.index.name = CLASS_COLUMN

class_statistics_df["instance_count"] = (
    class_instance_counts.reindex(
        all_classes,
        fill_value=0
    )
)

class_statistics_df["image_count"] = (
    class_image_counts.reindex(
        all_classes,
        fill_value=0
    )
)

# 每张相关图片平均出现多少个该类目标
class_statistics_df["instances_per_image"] = (
    class_statistics_df["instance_count"]
    / class_statistics_df["image_count"].replace(0, pd.NA)
)

# 占所有标注框的百分比
class_statistics_df["instance_percentage"] = (
    class_statistics_df["instance_count"]
    / num_instances
    * 100
)

# 占所有图像的百分比
class_statistics_df["image_percentage"] = (
    class_statistics_df["image_count"]
    / num_images
    * 100
)

# 是否理论上可以同时划入训练集和测试集
class_statistics_df["can_split_both_sides"] = (
    class_statistics_df["image_count"] >= 2
)


# ============================================================
# 5. 稀有程度分组
# ============================================================

def classify_rarity(image_count: int) -> str:
    if image_count == 0:
        return "missing"
    if image_count == 1:
        return "only_1_image"
    if image_count == 2:
        return "only_2_images"
    if image_count <= 5:
        return "3_to_5_images"
    if image_count <= 10:
        return "6_to_10_images"
    if image_count <= 50:
        return "11_to_50_images"
    return "more_than_50_images"


class_statistics_df["rarity_level"] = (
    class_statistics_df["image_count"]
    .apply(classify_rarity)
)


# ============================================================
# 6. 每张图像的统计
# ============================================================

image_instance_counts = (
    gt_df.groupby(FILENAME_COLUMN)
    .size()
)

image_class_counts = (
    gt_df.groupby(FILENAME_COLUMN)[CLASS_COLUMN]
    .nunique()
)

image_statistics_df = pd.DataFrame({
    "instance_count": image_instance_counts,
    "class_count": image_class_counts
})

image_statistics_df.index.name = FILENAME_COLUMN

# 标记是否为多类别图像
image_statistics_df["is_multi_class"] = (
    image_statistics_df["class_count"] > 1
)


# ============================================================
# 7. 数据集总体统计
# ============================================================

summary_data = {
    "total_images": num_images,
    "total_classes": num_classes,
    "total_instances": num_instances,
    "average_instances_per_image": (
        image_statistics_df["instance_count"].mean()
    ),
    "median_instances_per_image": (
        image_statistics_df["instance_count"].median()
    ),
    "max_instances_in_one_image": (
        image_statistics_df["instance_count"].max()
    ),
    "average_classes_per_image": (
        image_statistics_df["class_count"].mean()
    ),
    "median_classes_per_image": (
        image_statistics_df["class_count"].median()
    ),
    "max_classes_in_one_image": (
        image_statistics_df["class_count"].max()
    ),
    "single_class_images": (
        image_statistics_df["class_count"] == 1
    ).sum(),
    "multi_class_images": (
        image_statistics_df["class_count"] > 1
    ).sum(),
    "classes_in_only_one_image": (
        class_statistics_df["image_count"] == 1
    ).sum(),
    "classes_in_only_two_images": (
        class_statistics_df["image_count"] == 2
    ).sum(),
    "classes_in_3_to_5_images": (
        class_statistics_df["image_count"].between(3, 5)
    ).sum(),
    "classes_in_6_to_10_images": (
        class_statistics_df["image_count"].between(6, 10)
    ).sum(),
    "classes_in_more_than_10_images": (
        class_statistics_df["image_count"] > 10
    ).sum(),
}

dataset_summary_df = pd.DataFrame(
    list(summary_data.items()),
    columns=["metric", "value"]
)


# ============================================================
# 8. 保存 CSV
# ============================================================

class_statistics_path = (
    OUTPUT_DIR / "class_statistics.csv"
)

image_statistics_path = (
    OUTPUT_DIR / "image_statistics.csv"
)

dataset_summary_path = (
    OUTPUT_DIR / "dataset_summary.csv"
)

class_statistics_df.to_csv(
    class_statistics_path,
    encoding="utf-8-sig"
)

image_statistics_df.to_csv(
    image_statistics_path,
    encoding="utf-8-sig"
)

dataset_summary_df.to_csv(
    dataset_summary_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. 绘制每类实例数量柱状图
# ============================================================

plt.figure(figsize=(16, 7))

plt.bar(
    class_statistics_df.index.astype(str),
    class_statistics_df["instance_count"]
)

plt.xlabel("Class ID")
plt.ylabel("Number of Instances")
plt.title("Instance Distribution of 66 Traffic Sign Classes")

plt.xticks(
    rotation=90,
    fontsize=8
)

plt.tight_layout()

instance_plot_path = (
    OUTPUT_DIR / "class_instance_distribution.png"
)

plt.savefig(
    instance_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. 绘制每类出现图像数量柱状图
# ============================================================

plt.figure(figsize=(16, 7))

plt.bar(
    class_statistics_df.index.astype(str),
    class_statistics_df["image_count"]
)

plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.title("Image Distribution of 66 Traffic Sign Classes")

plt.xticks(
    rotation=90,
    fontsize=8
)

plt.tight_layout()

image_plot_path = (
    OUTPUT_DIR / "class_image_distribution.png"
)

plt.savefig(
    image_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. 绘制按实例数量排序的长尾图
# ============================================================

sorted_class_df = class_statistics_df.sort_values(
    by="instance_count",
    ascending=False
)

plt.figure(figsize=(16, 7))

plt.bar(
    sorted_class_df.index.astype(str),
    sorted_class_df["instance_count"]
)

plt.xlabel("Class ID, Sorted by Frequency")
plt.ylabel("Number of Instances")
plt.title("Long-Tailed Distribution of Traffic Sign Classes")

plt.xticks(
    rotation=90,
    fontsize=8
)

plt.tight_layout()

long_tail_plot_path = (
    OUTPUT_DIR / "class_long_tail_distribution.png"
)

plt.savefig(
    long_tail_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. 控制台输出重点类别
# ============================================================

print("\n===== 类别稀有程度统计 =====")
print(
    class_statistics_df["rarity_level"]
    .value_counts()
)

print("\n===== 只出现在1张图中的类别 =====")
only_one_image = class_statistics_df[
    class_statistics_df["image_count"] == 1
]

if only_one_image.empty:
    print("无")
else:
    print(
        only_one_image[
            [
                "instance_count",
                "image_count",
                "instances_per_image"
            ]
        ]
    )

print("\n===== 只出现在2张图中的类别 =====")
only_two_images = class_statistics_df[
    class_statistics_df["image_count"] == 2
]

if only_two_images.empty:
    print("无")
else:
    print(
        only_two_images[
            [
                "instance_count",
                "image_count",
                "instances_per_image"
            ]
        ]
    )

print("\n===== 实例数量最少的10个类别 =====")
print(
    class_statistics_df.sort_values(
        by="instance_count",
        ascending=True
    ).head(10)
)

print("\n===== 实例数量最多的10个类别 =====")
print(
    class_statistics_df.sort_values(
        by="instance_count",
        ascending=False
    ).head(10)
)

print("\n===== 输出完成 =====")
print(f"类别统计：{class_statistics_path.resolve()}")
print(f"图像统计：{image_statistics_path.resolve()}")
print(f"总体统计：{dataset_summary_path.resolve()}")
print(f"类别实例图：{instance_plot_path.resolve()}")
print(f"类别图像图：{image_plot_path.resolve()}")
print(f"长尾分布图：{long_tail_plot_path.resolve()}")