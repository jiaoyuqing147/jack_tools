import pandas as pd
import os

# 加载标注与尺寸文件
labels_path = r'F:\jack_dataset\EMTD\GT.csv'
sizes_path = r'image_sizes.csv'

df_labels = pd.read_csv(labels_path)
df_sizes = pd.read_csv(sizes_path)

# 合并尺寸信息
df = pd.merge(df_labels, df_sizes, on='filename', how='left')

# 检查是否有尺寸缺失
if df[['width', 'height']].isnull().any().any():
    raise ValueError("有些图片缺失尺寸信息，请检查尺寸表是否完整。")

# 输出文件夹
output_dir = 'labels_yolo'
os.makedirs(output_dir, exist_ok=True)

# 按图片分组
grouped = df.groupby('filename')

for filename, group in grouped:
    yolo_lines = []
    img_width = group.iloc[0]['width']
    img_height = group.iloc[0]['height']

    for _, row in group.iterrows():
        class_id = int(row['Class ID']) - 1  # YOLO类别从0开始
        x_center = ((row['xmin'] + row['xmax']) / 2) / img_width
        y_center = ((row['ymin'] + row['ymax']) / 2) / img_height
        width = (row['xmax'] - row['xmin']) / img_width
        height = (row['ymax'] - row['ymin']) / img_height

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)

    label_file = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')
    with open(label_file, 'w') as f:
        f.write('\n'.join(yolo_lines))

print(f"YOLO标注文件已生成在: {output_dir}")
