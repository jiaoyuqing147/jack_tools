import os
import pandas as pd

# ==== 配置路径 ====
detection_file = 'GT_Detection.txt'
recognition_file = 'GT_Recognition.txt'
image_size_file = 'image_sizes.csv'
labels_output_dir = 'labels_yolo'

os.makedirs(labels_output_dir, exist_ok=True)

# ==== 读取文件 ====
detection_df = pd.read_csv(detection_file, sep=';', engine='python')
recognition_df = pd.read_csv(recognition_file, sep=';', engine='python')
size_df = pd.read_csv(image_size_file)

# ==== 标准化文件名（小写、去除引号） ====
detection_df['File Name'] = detection_df['File Name'].str.strip("'").str.lower()
recognition_df['File Name'] = recognition_df['File Name'].str.strip("'").str.lower()
size_df['filename'] = size_df['filename'].str.lower()

# ==== 生成 scene 字段（识别）以对应 detection 文件名 ====
recognition_df['scene'] = recognition_df['File Name'].apply(lambda x: x.split('_')[0] + '.jpg')
recognition_grouped = recognition_df.groupby('scene')

# ==== 构建 size map ====
size_map = dict(zip(size_df['filename'], zip(size_df['width'], size_df['height'])))

missing_recog = []
missing_size = []
mismatches = []

for scene, group in detection_df.groupby('File Name'):
    if scene not in recognition_grouped.groups:
        missing_recog.append(scene)
        continue

    if scene not in size_map:
        missing_size.append(scene)
        continue

    recog_group = recognition_grouped.get_group(scene).reset_index(drop=True)

    if len(group) != len(recog_group):
        mismatches.append((scene, len(group), len(recog_group)))
        continue

    img_w, img_h = size_map[scene]
    yolo_labels = []

    for i, det_row in enumerate(group.itertuples(index=False)):
        recog_row = recog_group.iloc[i]

        class_id = recog_row['Class ID'] - 1  # YOLO 索引从0开始
        x_center = (det_row.X + det_row.Width / 2) / img_w
        y_center = (det_row.Y + det_row.Height / 2) / img_h
        width_norm = det_row.Width / img_w
        height_norm = det_row.Height / img_h

        label_line = f'{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}'
        yolo_labels.append(label_line)

    label_filename = os.path.splitext(scene)[0] + '.txt'
    label_path = os.path.join(labels_output_dir, label_filename)

    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_labels))

    print(f'[OK] Wrote YOLO label: {label_path}')

print('✅ 标签转换全部完成\n')

# ==== 输出统计信息 ====
if missing_recog:
    print(f'⚠ 缺少识别标注的图片数：{len(missing_recog)}')
    print(missing_recog)

if missing_size:
    print(f'⚠ 缺少图片尺寸信息的图片数：{len(missing_size)}')
    print(missing_size)

if mismatches:
    print(f'⚠ 检测与识别数量不符的图片数：{len(mismatches)}')
    for m in mismatches:
        print(f'Mismatch: {m[0]} - Detection: {m[1]}, Recognition: {m[2]}')

print('\n🎉 全部完成！')
