import os
import json
from PIL import Image

# 设置路径
json_path = r"D:\Jiao\dataset\traffic-light-detection-dataset\train_dataset\annotations\train.json"
image_dir = r"D:\Jiao\dataset\traffic-light-detection-dataset\train_dataset\train_images"
output_label_dir = r"D:\Jiao\dataset\traffic-light-detection-dataset\train_dataset\labels"

# 确保输出目录存在
os.makedirs(output_label_dir, exist_ok=True)

# 定义类别映射（假设 "traffic light" 类别 ID 为 0，灯光颜色可扩展）
class_mapping = {
    "traffic_light": 0,  # 交通灯整体框
    "red": 1,   # 红灯
    "green": 2,  # 绿灯
    "yellow": 3  # 黄灯
}

# 读取 JSON 文件
with open(json_path, "r") as f:
    data = json.load(f)

# 遍历 JSON 数据，转换为 YOLO 格式
for annotation in data["annotations"]:
    img_filename = annotation["filename"].split("\\")[-1]  # 提取文件名
    img_path = os.path.join(image_dir, img_filename)

    # 确保图片存在
    if not os.path.exists(img_path):
        print(f"⚠️ 图片 {img_filename} 不存在，跳过")
        continue

    # 读取实际图像尺寸
    with Image.open(img_path) as img:
        img_width, img_height = img.size

    label_filename = os.path.splitext(img_filename)[0] + ".txt"
    label_path = os.path.join(output_label_dir, label_filename)

    # 跳过 ignore == 1 的对象
    if annotation.get("ignore", 0) == 1:
        continue

    labels = []

    # 解析 bndbox（交通灯整体框）
    bbox = annotation["bndbox"]
    xmin, ymin, xmax, ymax = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]

    # 归一化坐标
    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymax) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height

    # 添加到标签列表（交通灯整体框类别 ID = 0）
    labels.append(f"{class_mapping['traffic_light']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 解析 inbox（交通灯内部灯光）
    for inbox_item in annotation.get("inbox", []):
        color = inbox_item["color"]
        inbox_bbox = inbox_item["bndbox"]
        xmin, ymin, xmax, ymax = inbox_bbox["xmin"], inbox_bbox["ymin"], inbox_bbox["xmax"], inbox_bbox["ymax"]

        # 归一化
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # 只添加已知颜色的灯光
        if color in class_mapping:
            labels.append(f"{class_mapping[color]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 保存 YOLO 格式文件
    with open(label_path, "w") as label_file:
        label_file.write("\n".join(labels) + "\n")

    print(f"✅ 生成标注: {label_filename}")

print("🚀 JSON 标注已转换为 YOLO 格式！")
