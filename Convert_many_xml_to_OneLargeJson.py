#本脚本运行成功，确实可以把针对每个图像的XML文件合并成为一个大的COCO格式的JSON文件。
import os
import json
import xml.etree.ElementTree as ET

# 设置VOC数据和输出的JSON文件的路径
voc_dir = 'W:/Jack_datasets/COCO/dataset/Jack_generate_cat/COCO/annotations/val2017'  # VOC XML文件的文件夹
output_json_file = 'W:/Jack_datasets/COCO/dataset/Jack_generate_cat/COCO/annotations/val2017_cat.json'  # 输出的COCO格式JSON文件


def xml_to_coco_json(xml_files, json_file):
    coco_data = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    # 这里假设你已经知道所有的类别和它们的ID
    category_map = {'cat': 1}  # 示例类别映射
    coco_data["categories"] = [{"id": v, "name": k} for k, v in category_map.items()]

    annotation_id = 1  # 初始化标注ID
    for i, xml_file in enumerate(xml_files, start=1):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 图像信息
        filename = root.find('filename').text
        size = root.find('size')
        image_info = {
            "id": i,
            "file_name": filename,
            "width": int(size.find('width').text),
            "height": int(size.find('height').text)
        }
        coco_data["images"].append(image_info)

        # 对象标注信息
        for obj in root.findall('object'):
            category = obj.find('name').text
            category_id = category_map.get(category, None)
            bndbox = obj.find('bndbox')
            annotation_info = {
                "id": annotation_id,
                "image_id": i,
                "category_id": category_id,
                "bbox": [
                    int(bndbox.find('xmin').text),
                    int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text),
                    int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text)
                ],
                "area": (int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text)) *
                        (int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text)),
                "segmentation": [],
                "iscrowd": 0
            }
            coco_data["annotations"].append(annotation_info)
            annotation_id += 1

    with open(json_file, 'w') as f:
        json.dump(coco_data, f, indent=4)


# 获取VOC XML文件列表
xml_files = [os.path.join(voc_dir, x) for x in os.listdir(voc_dir) if x.endswith('.xml')]
xml_to_coco_json(xml_files, output_json_file)
