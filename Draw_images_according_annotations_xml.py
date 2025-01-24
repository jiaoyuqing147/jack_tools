import cv2
import xml.etree.ElementTree as ET

# 读取 XML 文件并解析边界框信息
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = root.findall('object')
    bboxes = []
    for obj in objects:
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bboxes.append((xmin, ymin, xmax, ymax))
    return bboxes

# 在图像上绘制边界框
def draw_bboxes(image_path, bboxes):
    image = cv2.imread(image_path)
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return image

# XML 文件路径
xml_file = 'W:/Jack_datasets/COCO/dataset/Jack_generate_cat/COCO/annotations/train2017/000000000977.xml'  # 替换为实际的 XML 文件路径

# 图像文件路径
image_path = 'W:/Jack_datasets/COCO/dataset/Jack_generate_cat/COCO/images/train2017/000000000977.jpg'  # 替换为实际的图像文件路径

# 解析 XML 文件中的边界框信息
bboxes = parse_xml(xml_file)

# 在图像上绘制边界框
image_with_bboxes = draw_bboxes(image_path, bboxes)

# 显示图像
cv2.imshow('Image with Bounding Boxes', image_with_bboxes)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 如果需要保存绘制结果，可以使用以下代码
# output_image_path = 'path/to/output_image.jpg'  # 替换为实际的输出路径
# cv2.imwrite(output_image_path, image_with_bboxes)
