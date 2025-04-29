import os
import xml.etree.ElementTree as ET
import cv2
#可视化交通信号灯用这个的
# 数据集路径
# image_dir = r"D:\Jiao\dataset\S2TLD\S2TLD7201280\normal_2\JPEGImages"
image_dir = r"W:\Jack_datasets\S2TLD\S2TLD\S2TLD720x1280\normal_2\JPEGImages"
# xml_dir = r"D:\Jiao\dataset\S2TLD\S2TLD7201280\normal_2\Annotations"
xml_dir = r"W:\Jack_datasets\S2TLD\S2TLD\S2TLD720x1280\normal_2\Annotations"
output_dir = r"W:\Jack_datasets\S2TLD\S2TLD\S2TLD720x1280\normal_2\visualized_labels"

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 类别颜色映射
color_map = {
    "red": (0, 0, 255),  # 红色
    "yellow": (0, 255, 255),  # 黄色
    "green": (0, 255, 0),  # 绿色
    "off": (200, 200, 200)  # 灰色
}

# 遍历 XML 标注文件
for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(xml_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图片文件名
    image_filename = root.find("filename").text
    image_path = os.path.join(image_dir, image_filename)

    if not os.path.exists(image_path):
        print(f"⚠️ 图片 {image_filename} 不存在，跳过")
        continue

    # 读取图片
    print(f"尝试读取图片: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 读取失败: {image_path}")
        continue
    # 解析目标框
    for obj in root.findall("object"):
        class_name = obj.find("name").text.lower()
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        # 颜色
        color = color_map.get(class_name, (255, 255, 255))  # 未知类别用白色

        # 绘制矩形框
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, class_name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 保存带有标注的图像
    output_path = os.path.join(output_dir, image_filename)
    cv2.imwrite(output_path, image)
    print(f"✅ 生成可视化标注: {output_path}")

print("🚀 所有标注已绘制完成！")
