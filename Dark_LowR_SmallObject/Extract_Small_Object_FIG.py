import os
import shutil
import xml.etree.ElementTree as ET

# 定义路径
image_folder = r"D:\Jiao\dataset\CatOnlyCOCOVOC\Original\images"
xml_folder = r"D:\Jiao\dataset\CatOnlyCOCOVOC\Original\annotations\xmlall"

output_image_folder = r"D:\Jiao\dataset\CatOnlyCOCOVOC\SmallTarget\images"
output_xml_folder = r"D:\Jiao\dataset\CatOnlyCOCOVOC\SmallTarget\annotations"

# 确保输出目录存在
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_xml_folder, exist_ok=True)

# 小目标阈值
SMALL_OBJECT_AREA = 32 * 32  # 1024 像素

# 统计信息
total_images = 0
small_target_images = 0

# 遍历 XML 文件
for xml_file in os.listdir(xml_folder):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(xml_folder, xml_file)

    # 解析 XML 文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 读取图像的宽高
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    has_small_object = False

    # 遍历目标对象
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # 计算目标面积
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        area = bbox_width * bbox_height

        # 判断是否是小目标
        if area < SMALL_OBJECT_AREA:
            has_small_object = True
            break  # 只要有一个小目标，就处理该图片

    # 如果包含小目标，则复制文件
    if has_small_object:
        small_target_images += 1
        img_filename = xml_file.replace(".xml", ".jpg")  # 假设图片是 JPG 格式
        img_path = os.path.join(image_folder, img_filename)

        # 确保图片存在
        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(output_image_folder, img_filename))
            shutil.copy(xml_path, os.path.join(output_xml_folder, xml_file))

    total_images += 1

print(f"总处理图片数: {total_images}")
print(f"包含小目标的图片数: {small_target_images}")
print("筛选完成！🚀")
