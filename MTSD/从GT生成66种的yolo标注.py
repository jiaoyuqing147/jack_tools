import pandas as pd
import os

# ==== 配置路径 ====
gt_file = 'GT.txt'
size_file = 'image_sizes.csv'
output_dir = 'labels_yolo_66classes'

os.makedirs(output_dir, exist_ok=True)

# ==== 读取标注和尺寸 ====
gt_cols = ['File Name', 'X', 'Y', 'Width', 'Height', 'Sign Type', 'Sign Group', 'Sign Class',
           'TS Class', 'TS ID', 'TS Color', 'Shape', 'Shape ID', 'Lightning', 'Image Source']
gt_df = pd.read_csv(gt_file, delimiter=';', encoding='latin1', header=None, names=gt_cols)

size_df = pd.read_csv(size_file)

# 标准化文件名（小写，去引号，去空格）
gt_df['File Name'] = gt_df['File Name'].str.strip().str.strip("'").str.lower()
size_df['filename'] = size_df['filename'].str.strip().str.strip("'").str.lower()

# 转换坐标和尺寸为数值型
gt_df['X'] = pd.to_numeric(gt_df['X'], errors='coerce')
gt_df['Y'] = pd.to_numeric(gt_df['Y'], errors='coerce')
gt_df['Width'] = pd.to_numeric(gt_df['Width'], errors='coerce')
gt_df['Height'] = pd.to_numeric(gt_df['Height'], errors='coerce')
gt_df['TS ID'] = pd.to_numeric(gt_df['TS ID'], errors='coerce')

# 文件名到尺寸映射
size_dict = {row['filename']: (row['width'], row['height']) for _, row in size_df.iterrows()}

# ==== 按文件分组，生成YOLO格式 ====
for filename, group in gt_df.groupby('File Name'):
    if filename not in size_dict:
        print(f"⚠️ 尺寸缺失，跳过 {filename}")
        continue

    img_w, img_h = size_dict[filename]
    yolo_lines = []

    for _, row in group.iterrows():
        if pd.isnull(row['X']) or pd.isnull(row['Width']) or pd.isnull(row['TS ID']):
            continue  # 跳过有缺失的行

        x_center = (row['X'] + row['Width'] / 2) / img_w
        y_center = (row['Y'] + row['Height'] / 2) / img_h
        w_norm = row['Width'] / img_w
        h_norm = row['Height'] / img_h
        class_id = int(row['TS ID']) - 1  # TS ID 1~66 -> YOLO类别从0开始

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        yolo_lines.append(yolo_line)

    label_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')
    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

    print(f"✅ 生成：{label_path}")

print(f"\n所有YOLO格式66类标注已保存至：{output_dir}")
