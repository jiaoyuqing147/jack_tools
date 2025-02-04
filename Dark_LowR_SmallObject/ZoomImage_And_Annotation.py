import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# 输入 & 输出路径
# image_dir = r"F:\jack_dataset\cocoalldata\Jack_generate_cat\COCO\dark\val2017Dark"  # 原始暗光图像路径
# xml_dir = r"F:\jack_dataset\cocoalldata\Jack_generate_cat\COCO\dark\val2017Dark_annotations"  # 对应的 XML 标注文件路径
# output_image_dir = r"F:\jack_dataset\cocoalldata\Jack_generate_cat\COCO\dark\val2017Dark_resize"  # 目标存储缩放后图像的文件夹
# output_xml_dir = r"F:\jack_dataset\cocoalldata\Jack_generate_cat\COCO\dark\val2017Dark_annotations_resize"  # 目标存储修改后 XML 的文件夹


image_dir = r"D:\Jiao\dataset\CatOnlyCOCOVOC\Dark\images"  # 原始暗光图像路径
xml_dir = r"D:\Jiao\dataset\CatOnlyCOCOVOC\Dark\annotations"  # 对应的 XML 标注文件路径
output_image_dir = r"D:\Jiao\dataset\CatOnlyCOCOVOC\DarkResize300\images"  # 目标存储缩放后图像的文件夹
output_xml_dir = r"D:\Jiao\dataset\CatOnlyCOCOVOC\DarkResize300\annotations"  # 目标存储修改后 XML 的文件夹

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_xml_dir, exist_ok=True)

# 目标最大尺寸
MAX_SIZE = 300  # 限制最大宽或高不超过 300


# 添加高斯噪声
def add_weak_gaussian_noise(image, mean=0, stddev=3, prob=0.3):
    """减少高斯噪声影响，确保更真实"""
    # if np.random.rand() > prob:
    #     return image  # 以一定概率不加噪声

    noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)  # 用 float 计算
    noisy_image = image.astype(np.float32) + noise  # 避免 uint8 截断

    # 确保 noisy_image 是 numpy.ndarray 类型
    noisy_image = np.array(noisy_image, dtype=np.float32)

    # 限制到 0-255 并转换为 uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image



# 添加泊松噪声
def add_poisson_noise(image):
    """给图片添加泊松噪声"""
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))  # 确保像素值分布
    noisy_image = np.random.poisson(image * vals) / float(vals)

#
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
    #resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)#这样压缩出阿里的图像质量较高
    #resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)#这样压缩出来的图像质量较低
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)#双三次插值 (Bicubic Interpolation)

    # 添加噪声（可以选择高斯噪声或泊松噪声）不加了，感觉加了噪声效果很差
    # noisy_image = add_weak_gaussian_noise(resized_image, mean=0, stddev=10)  # 高斯噪声
    # noisy_image = add_poisson_noise(resized_image)  # 泊松噪声（替换上面这行）

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
