# import pandas as pd
# import os
#
# # ==== 配置路径 ====
# gt_file = 'GT_Detection.txt'
# size_file = 'image_sizes.csv'
# output_dir = 'labels_yolo'
#
# os.makedirs(output_dir, exist_ok=True)
#
# # ==== 读取标注和尺寸数据 ====
# gt_df = pd.read_csv(gt_file, delimiter=';', encoding='latin1')
# size_df = pd.read_csv(size_file)
#
# # ✅ 文件名统一小写 strip
# gt_df['File Name'] = gt_df['File Name'].str.strip().str.strip("'").str.lower()
# size_df['filename'] = size_df['filename'].str.strip().str.strip("'").str.lower()
#
# # 建立尺寸映射
# size_dict = {row['filename']: (row['width'], row['height']) for _, row in size_df.iterrows()}
#
# # ==== 按文件分组并写入 yolo 格式 ====
# for filename, group in gt_df.groupby('File Name'):
#     if filename not in size_dict:
#         print(f"⚠️ 跳过：{filename}，因尺寸信息缺失")
#         continue
#
#     img_w, img_h = size_dict[filename]
#     yolo_lines = []
#     for _, row in group.iterrows():
#         x_center = (row['X'] + row['Width'] / 2) / img_w
#         y_center = (row['Y'] + row['Height'] / 2) / img_h
#         w_norm = row['Width'] / img_w
#         h_norm = row['Height'] / img_h
#         class_id = row['Class ID']
#
#         yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
#         yolo_lines.append(yolo_line)
#
#     label_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')
#     with open(label_path, 'w') as f:
#         f.write('\n'.join(yolo_lines))
#
#     print(f"✅ 生成：{label_path}")
#
# print(f"\n所有标注已保存至：{output_dir}")
#
#
#
#
# 上面的是粗类别，10个类别
#
import pandas as pd
import os

# ==== 配置路径 ====
gt_recognition_file = 'GT_Recognition.txt'
size_file = 'image_sizes.csv'
output_dir = 'labels_yolo_recognition'

os.makedirs(output_dir, exist_ok=True)

# ==== 读取标注和尺寸数据 ====
gt_df = pd.read_csv(gt_recognition_file, delimiter=';', encoding='latin1', header=None)
gt_df.columns = ['File Name', 'Sign Type', 'Sign Group', 'Sign Class', 'TS Class', 'Class ID', 'Lightning', 'Image Source']

size_df = pd.read_csv(size_file)

# 标准化文件名（小写，去除引号）
gt_df['File Name'] = gt_df['File Name'].str.strip().str.strip("'").str.lower()
size_df['filename'] = size_df['filename'].str.strip().str.strip("'").str.lower()

# 构建文件名到尺寸的映射
size_dict = {row['filename']: (row['width'], row['height']) for _, row in size_df.iterrows()}

# ==== 按文件生成 yolo 标注 ====
for _, row in gt_df.iterrows():
    filename = row['File Name']
    class_id = row['Class ID']  # 66类标注

    if filename not in size_dict:
        print(f"⚠️ 尺寸缺失，跳过 {filename}")
        continue

    img_w, img_h = size_dict[filename]

    # 识别任务的图片是已裁剪好的单目标图片，因此中心点是图像中心，宽高归一化都是1
    x_center, y_center, w_norm, h_norm = 0.5, 0.5, 1.0, 1.0

    yolo_label = f"{class_id} {x_center} {y_center} {w_norm} {h_norm}"

    label_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')
    with open(label_path, 'w') as f:
        f.write(yolo_label)

    print(f"✅ 生成：{label_path}")

print(f"\n所有识别标注已保存到：{output_dir}")
