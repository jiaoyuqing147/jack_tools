import os
#不要用官网给的Train文件夹，那些图像太少了600张，不如用full文件夹的，900张，其中741张都是有标注的
# 配置
gt_txt_path = r'/home/jiaoyuqing/bigspace/workspaceJack/datasets/FullIJCNN2013/FullIJCNN2013/gt.txt'  # gt.txt 路径
images_dir = r'/home/jiaoyuqing/bigspace/workspaceJack/datasets/FullIJCNN2013/FullIJCNN2013/'  # 图片所在目录
labels_dir = r'/home/jiaoyuqing/bigspace/workspaceJack/datasets/FullIJCNN2013/FullIJCNN2013/labels'  # 输出的标签txt目录
# gt_txt_path = r'E:\DataSets\FullIJCNN2013\gt.txt'  # gt.txt 路径
# images_dir = r'E:\DataSets\FullIJCNN2013'  # 图片所在目录
# labels_dir = r'E:\DataSets\FullIJCNN2013\labels'  # 输出的标签txt目录
#


IMAGE_WIDTH = 1360  # GTSDB图像固定宽度
IMAGE_HEIGHT = 800  # GTSDB图像固定高度

os.makedirs(labels_dir, exist_ok=True)

with open(gt_txt_path, 'r') as f:
    lines = f.readlines()

labels_dict = {}

for line in lines:
    parts = line.strip().split(';')
    if len(parts) < 6:
        continue

    image_name = parts[0].replace('.ppm', '')
    x_min = int(parts[1])
    y_min = int(parts[2])
    x_max = int(parts[3])
    y_max = int(parts[4])
    class_id = int(parts[5])

    # 计算中心点和宽高，归一化
    x_center = ((x_min + x_max) / 2) / IMAGE_WIDTH
    y_center = ((y_min + y_max) / 2) / IMAGE_HEIGHT
    width = (x_max - x_min) / IMAGE_WIDTH
    height = (y_max - y_min) / IMAGE_HEIGHT

    label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

    if image_name not in labels_dict:
        labels_dict[image_name] = []
    labels_dict[image_name].append(label_line)

# 写入每张图片对应的txt文件
for image_name, label_lines in labels_dict.items():
    label_path = os.path.join(labels_dir, f'{image_name}.txt')
    with open(label_path, 'w') as f:
        f.writelines(label_lines)

print(f"YOLO格式标签已生成在目录：{labels_dir}")
