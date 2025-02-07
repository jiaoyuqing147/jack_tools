import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# COCO 2017 交通灯类别 ID（Traffic Light 类别 ID = 10）
TRAFFIC_LIGHT_ID = 10


def load_coco_annotations(json_path):
    """ 加载 COCO 标注 JSON 文件 """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_traffic_light_data(coco_data):
    """ 提取 COCO 数据集中所有交通灯目标的尺寸信息 """
    traffic_light_sizes = []
    traffic_light_ratios = []
    image_sizes = {}

    # 记录图片的宽高
    for img in coco_data['images']:
        image_sizes[img['id']] = (img['width'], img['height'])

    # 遍历所有标注信息
    for ann in coco_data['annotations']:
        if ann['category_id'] == TRAFFIC_LIGHT_ID:
            image_id = ann['image_id']
            img_width, img_height = image_sizes[image_id]

            x, y, w, h = ann['bbox']  # COCO bbox 格式: [x, y, width, height]
            area = w * h
            img_area = img_width * img_height

            # 计算尺寸比例
            size_ratio = area / img_area
            aspect_ratio = w / h

            traffic_light_sizes.append(size_ratio)
            traffic_light_ratios.append(aspect_ratio)

    return traffic_light_sizes, traffic_light_ratios


def plot_distribution(data, title, xlabel):
    """ 绘制数据分布图 """
    plt.figure(figsize=(8, 5))
    sns.histplot(data, bins=50, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.show()


# ========== 运行分析 ========== #
coco_json_path = "instances_train2017.json"  # 替换为你的 JSON 文件路径
coco_data = load_coco_annotations(coco_json_path)
traffic_light_sizes, traffic_light_ratios = extract_traffic_light_data(coco_data)

# 绘制交通灯目标面积占比的分布
plot_distribution(traffic_light_sizes, "Traffic Light Size Distribution", "Size Ratio (BBox Area / Image Area)")

# 绘制交通灯的宽高比分布
plot_distribution(traffic_light_ratios, "Traffic Light Aspect Ratio Distribution", "Aspect Ratio (Width / Height)")

# 计算小目标占比
small_object_threshold = 0.05  # 5% 面积比例
small_object_ratio = sum(np.array(traffic_light_sizes) < small_object_threshold) / len(traffic_light_sizes)
print(f"小目标（面积 < 5%）占比: {small_object_ratio * 100:.2f}%")
