import pandas as pd
import os

# ==== 配置路径 ====
gt_file = 'GT.csv'
size_file = 'image_sizes.csv'
output_dir = 'labels_yolo_66classes'

# ==== 加载标注和图片尺寸 ====
df_gt = pd.read_csv(gt_file)
df_sizes = pd.read_csv(size_file)

# 标准化文件名（全小写避免匹配错误）
df_gt['filename'] = df_gt['filename'].str.lower()
df_sizes['filename'] = df_sizes['filename'].str.lower()

# 合并尺寸信息
df = pd.merge(df_gt, df_sizes, on='filename', how='left')

# 检查是否有图片没有尺寸
missing_sizes = df[df[['width', 'height']].isnull().any(axis=1)]
if not missing_sizes.empty:
    print("以下图片缺失尺寸信息:")
    print(missing_sizes['filename'].unique())
    raise ValueError("部分图片缺尺寸，终止。请检查 image_sizes.csv 是否完整。")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 按图片分组生成 YOLO txt
for filename, group in df.groupby('filename'):
    img_w = group.iloc[0]['width']
    img_h = group.iloc[0]['height']
    yolo_lines = []

    for _, row in group.iterrows():
        class_id = int(row['Class ID']) - 1  # YOLO从0开始

        # 提取原始坐标
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

        # 计算中心点和宽高 (归一化)
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)

    # 写文件
    label_file = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')
    with open(label_file, 'w') as f:
        f.write('\n'.join(yolo_lines))

print(f"YOLO格式标注已全部保存至：{output_dir}")
