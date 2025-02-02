import os
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# 处理后的图片 & XML 目录
image_dir = r"D:\Jiao\dataset\Jack_generate_cat\lowQulityDark\images\val2017Dark_resize"  # 替换为你的缩放后图像路径
xml_dir = r"D:\Jiao\dataset\Jack_generate_cat\lowQulityDark\annotations\val2017Dark_resize_annotations"  # 替换为你的修改后 XML 标注路径,不能是txt

def visualize_image_with_bbox(image_path, xml_path):
    # 读取图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV 默认 BGR，转换为 RGB

    # 解析 XML 文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取目标框
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # 画出目标框
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # 蓝色框

    # 显示图像
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")  # 隐藏坐标轴
    plt.show()

# 随机选择一个文件进行可视化
sample_image = os.listdir(image_dir)[50]  # 选取第一个图像
sample_xml = sample_image.replace(".jpg", ".xml").replace(".png", ".xml").replace(".jpeg", ".xml")


image_path = os.path.join(image_dir, sample_image)
xml_path = os.path.join(xml_dir, sample_xml)

visualize_image_with_bbox(image_path, xml_path)
