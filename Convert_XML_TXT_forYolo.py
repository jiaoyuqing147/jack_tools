import os
import xml.etree.ElementTree as ET

# 设置路径
xml_dir = r"D:\Jiao\dataset\S2TLD\S2TLD7201280\normal_1\Annotations"
image_dir = r"D:\Jiao\dataset\S2TLD\S2TLD7201280\normal_1\JPEGImages"
output_label_dir = r"D:\Jiao\dataset\S2TLD\S2TLD7201280\normal_1\labels"

# 确保输出目录存在
os.makedirs(output_label_dir, exist_ok=True)

# 类别映射（YOLO 需要 class_id）
class_mapping = {
    "red": 0,
    "yellow": 1,
    "green": 2,
    "off": 3
}

# 遍历 XML 文件
for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(xml_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图片信息
    filename = root.find("filename").text
    img_path = os.path.join(image_dir, filename)
    label_filename = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(output_label_dir, label_filename)

    # 读取图片尺寸
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    labels = []

    # 解析每个目标
    for obj in root.findall("object"):
        class_name = obj.find("name").text.lower()
        if class_name not in class_mapping:
            print(f"⚠️ 未知类别: {class_name}，跳过")
            continue

        class_id = class_mapping[class_name]
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # 归一化 YOLO 格式
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 保存到 YOLO 格式的 TXT
    if labels:
        with open(label_path, "w") as f:
            f.write("\n".join(labels) + "\n")

    print(f"✅ 生成 YOLO 标注: {label_filename}")

print("🚀 XML 转换 YOLO 格式完成！")
