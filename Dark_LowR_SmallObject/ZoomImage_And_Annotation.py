import os
import cv2
import xml.etree.ElementTree as ET

# 输入 & 输出路径
image_dir = r"F:\jack_dataset\cocoalldata\Jack_generate_cat\COCO\dark\val2017Dark"  # 原始暗光图像路径
xml_dir = r"F:\jack_dataset\cocoalldata\Jack_generate_cat\COCO\dark\val2017Dark_annotations"  # 对应的 XML 标注文件路径
output_image_dir = r"F:\jack_dataset\cocoalldata\Jack_generate_cat\COCO\dark\val2017Dark_resize"  # 目标存储缩放后图像的文件夹
output_xml_dir = r"F:\jack_dataset\cocoalldata\Jack_generate_cat\COCO\dark\val2017Dark_annotations_resize"  # 目标存储修改后 XML 的文件夹
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_xml_dir, exist_ok=True)

# 目标最大尺寸
MAX_SIZE = 300  # 限制最大宽或高不超过 300


def resize_image_and_update_xml(image_path, xml_path, output_img_path, output_xml_path):
    # 读取 XML 标注文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取原始图像尺寸
    size_tag = root.find("size")
    orig_width = int(size_tag.find("width").text)
    orig_height = int(size_tag.find("height").text)

    # 计算缩放比例，保持宽高比
    scale = min(MAX_SIZE / orig_width, MAX_SIZE / orig_height)
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    # 读取并缩放图像
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_img_path, resized_image)

    # 更新 XML 文件中的尺寸信息
    size_tag.find("width").text = str(new_width)
    size_tag.find("height").text = str(new_height)

    # 处理 XML 中的目标框坐标
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # 按照相同比例缩放目标框
        bndbox.find("xmin").text = str(int(xmin * scale))
        bndbox.find("ymin").text = str(int(ymin * scale))
        bndbox.find("xmax").text = str(int(xmax * scale))
        bndbox.find("ymax").text = str(int(ymax * scale))

    # 保存修改后的 XML 文件
    tree.write(output_xml_path)


# 遍历所有图像和 XML 文件
for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(image_dir, filename)
        xml_path = os.path.join(xml_dir,
                                filename.replace(".jpg", ".xml").replace(".png", ".xml").replace(".jpeg", ".xml"))

        if os.path.exists(xml_path):
            output_img_path = os.path.join(output_image_dir, filename)
            output_xml_path = os.path.join(output_xml_dir,
                                           filename.replace(".jpg", ".xml").replace(".png", ".xml").replace(".jpeg",
                                                                                                            ".xml"))
            resize_image_and_update_xml(image_path, xml_path, output_img_path, output_xml_path)
            print(f"✅ 处理完成: {filename}")

print("🎯 所有图片和标注已成功缩放并保存！")
