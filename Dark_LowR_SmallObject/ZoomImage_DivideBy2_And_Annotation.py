'''
📌 代码解读
遍历 Annotations/ 文件夹中的所有 XML。
解析 XML，获取图像尺寸，并缩小 1/2。
修改 size 和 bndbox 坐标（目标框），使其也缩小 1/2。
复制并缩小对应的 JPEG 图像。
保存新的 XML 到 Annotations_LR/，新的低分辨率图片到 JPEGImages_LR/。
'''


import os
import xml.etree.ElementTree as ET
import cv2

# 数据路径
hr_image_dir = r"D:\Jiao\dataset\CatOnlyCOCOVOC\Dark\images"      # 高分辨率图片目录
lr_image_dir = r"D:\Jiao\dataset\CatOnlyCOCOVOC\DarkResizeDivideBy2\images"   # 低分辨率图片存放目录
hr_xml_dir = r"D:\Jiao\dataset\CatOnlyCOCOVOC\Dark\annotations"       # 高分辨率 XML 目录
lr_xml_dir = r"D:\Jiao\dataset\CatOnlyCOCOVOC\DarkResizeDivideBy2\annotations"    # 低分辨率 XML 存放目录

# 确保输出目录存在
os.makedirs(lr_image_dir, exist_ok=True)
os.makedirs(lr_xml_dir, exist_ok=True)

# 设定缩小比例（1/2 缩小）
scale_factor = 0.5

# 遍历所有 XML 文件
for xml_file in os.listdir(hr_xml_dir):
    if not xml_file.endswith(".xml"):
        continue

    # 解析 XML 文件
    xml_path = os.path.join(hr_xml_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取原始图像尺寸
    size = root.find("size")
    orig_width = int(size.find("width").text)
    orig_height = int(size.find("height").text)

    # 计算新的图像尺寸
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)

    # 读取对应的高分辨率图像
    image_filename = root.find("filename").text
    hr_image_path = os.path.join(hr_image_dir, image_filename)
    lr_image_path = os.path.join(lr_image_dir, image_filename)

    # 读取并缩放图像
    image = cv2.imread(hr_image_path)
    if image is None:
        print(f"无法读取图像 {hr_image_path}，跳过...")
        continue

    image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(lr_image_path, image_resized)  # 保存低分辨率图像

    # 更新 XML 中的图像尺寸
    size.find("width").text = str(new_width)
    size.find("height").text = str(new_height)

    # 遍历所有目标，调整 bounding box
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(int(bbox.find("xmin").text) * scale_factor)
        ymin = int(int(bbox.find("ymin").text) * scale_factor)
        xmax = int(int(bbox.find("xmax").text) * scale_factor)
        ymax = int(int(bbox.find("ymax").text) * scale_factor)

        # 更新 bounding box 坐标
        bbox.find("xmin").text = str(xmin)
        bbox.find("ymin").text = str(ymin)
        bbox.find("xmax").text = str(xmax)
        bbox.find("ymax").text = str(ymax)

    # 保存调整后的 XML
    new_xml_path = os.path.join(lr_xml_dir, xml_file)
    tree.write(new_xml_path)

print(f"所有图像和 XML 文件已调整并保存至 {lr_image_dir} 和 {lr_xml_dir}")
