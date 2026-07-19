import os
import pandas as pd


GT_CSV_PATH = "GT.csv"
OUTPUT_DIR = "distribution_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

gt_df = pd.read_csv(GT_CSV_PATH)

required_columns = {"filename", "Class ID"}
missing_columns = required_columns - set(gt_df.columns)

if missing_columns:
    raise ValueError(f"GT.csv 缺少字段：{sorted(missing_columns)}")

gt_df = gt_df.dropna(subset=["filename", "Class ID"]).copy()

gt_df["filename"] = (
    gt_df["filename"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# 每类实例数
instance_counts = (
    gt_df["Class ID"]
    .value_counts()
    .sort_index()
)

# 每类出现的图像数
image_counts = (
    gt_df[["filename", "Class ID"]]
    .drop_duplicates()
    ["Class ID"]
    .value_counts()
    .sort_index()
)

distribution_df = pd.DataFrame({
    "instance_count": instance_counts,
    "image_count": image_counts
}).fillna(0).astype(int)

distribution_df["instances_per_image"] = (
    distribution_df["instance_count"]
    / distribution_df["image_count"].replace(0, pd.NA)
)

# 理论上是否可以在训练集和测试集两边都出现
distribution_df["can_split_both_sides"] = (
    distribution_df["image_count"] >= 2
)

# 稀有程度
def rarity_level(image_count):
    if image_count == 1:
        return "only_1_image"
    elif image_count == 2:
        return "only_2_images"
    elif image_count <= 5:
        return "3_to_5_images"
    elif image_count <= 10:
        return "6_to_10_images"
    else:
        return "more_than_10_images"

distribution_df["rarity_level"] = (
    distribution_df["image_count"].apply(rarity_level)
)

# 每张图包含的类别数和实例数
classes_per_image = (
    gt_df.groupby("filename")["Class ID"]
    .nunique()
)

instances_per_image = (
    gt_df.groupby("filename")
    .size()
)

image_summary_df = pd.DataFrame({
    "class_count": classes_per_image,
    "instance_count": instances_per_image
})

print("===== 数据集总体情况 =====")
print(f"图像总数：{gt_df['filename'].nunique()}")
print(f"类别总数：{gt_df['Class ID'].nunique()}")
print(f"标注框总数：{len(gt_df)}")
print(f"平均每张图实例数：{instances_per_image.mean():.2f}")
print(f"平均每张图类别数：{classes_per_image.mean():.2f}")

print("\n===== 类别稀有程度 =====")
print(distribution_df["rarity_level"].value_counts())

print("\n===== 只出现在1张图中的类别 =====")
print(
    distribution_df[
        distribution_df["image_count"] == 1
    ]
)

print("\n===== 只出现在2张图中的类别 =====")
print(
    distribution_df[
        distribution_df["image_count"] == 2
    ]
)

print("\n===== 完整类别分布 =====")
with pd.option_context("display.max_rows", None):
    print(distribution_df)

distribution_df.to_csv(
    os.path.join(OUTPUT_DIR, "class_distribution.csv"),
    index_label="Class ID"
)

image_summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "image_distribution.csv"),
    index_label="filename"
)

print("\n✅ 分析结果已保存到 distribution_analysis 文件夹")