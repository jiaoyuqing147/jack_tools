from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw


# 包含所有类别的原coco数据集路径
'''
目录格式如下：
$COCO_PATH
----|annotations
----|train2017
----|val2017
----|test2017
'''

headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''


# 检查目录是否存在，如果存在，先删除再创建，否则，直接创建
def mkr(path):
    if not os.path.exists(path):
        os.makedirs(path)  # 可以创建多级目录


def id2name(coco):
    classes = dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']
    return classes


def write_xml(anno_path, head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        f.write(objstr % (obj[0], obj[1], obj[2], obj[3], obj[4]))
    f.write(tail)


def save_annotations_and_imgs(coco, dataset, filename, objs):
    # 将图片转为xml，例:COCO_train2017_000000196610.jpg-->COCO_train2017_000000196610.xml
    dst_anno_dir = os.path.join(anno_dir, dataset)
    mkr(dst_anno_dir)
    anno_path = dst_anno_dir + '/' + filename[:-3] + 'xml'
    img_path = dataDir + dataset + '/' + filename
    print("img_path: ", img_path)

    if not os.path.exists(img_path):
        print(f"Image path does not exist: {img_path}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return

    dst_img_dir = os.path.join(img_dir, dataset)
    mkr(dst_img_dir)
    dst_imgpath = dst_img_dir + '/' + filename
    print("dst_imgpath: ", dst_imgpath)
    img = cv2.imread(img_path)
    # if (img.shape[2] == 1):
    #    print(filename + " not a RGB image")
    #   return
    shutil.copy(img_path, dst_imgpath)

    head = headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    write_xml(anno_path, head, objs, tail)


def showimg(coco, dataset, img, classes, cls_id, show=True):
    global dataDir
    I = Image.open('%s/%s/%s' % (dataDir, dataset, img['file_name']))
    # 通过id，得到注释的信息
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        class_name = classes[ann['category_id']]
        if class_name in classes_names:
            print(class_name)
            if 'bbox' in ann:
                bbox = ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [class_name, xmin, ymin, xmax, ymax]
                objs.append(obj)
                draw = ImageDraw.Draw(I)
                draw.rectangle([xmin, ymin, xmax, ymax])
    if show:
        plt.figure()
        plt.axis('off')
        plt.imshow(I)
        plt.show()

    return objs


def process_dataset(dataset, classes_names, dataDir, showimg, save_annotations_and_imgs):
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataset)
    coco = COCO(annFile) # 使用COCO API用来初始化注释数据
    classes = id2name(coco) # 获取COCO数据集中的所有类别
    print(classes)
    print(len(classes))
    classes_ids = coco.getCatIds(catNms=classes_names)#根据类别的名称name来获取类别的编号，这个编号是个列表形式，尽管只取一个类别，也是列表
    print(classes_ids)
    # 获取该类的id
    for cls in classes_names:
        cls_id = coco.getCatIds(catNms=[cls])#获取这个类别的id编号，如person就是 [1]
        img_ids = coco.getImgIds(catIds=cls_id)#获取含有这个id编号的图片的所有编号
        print(cls, len(img_ids))

        for imgId in tqdm(img_ids):
            img = coco.loadImgs(imgId)[0]
            filename = img['file_name']
            objs = showimg(coco, dataset, img, classes, classes_ids, show=False)#showing函数获取了img这幅图中每个目标的坐标
            print(objs)#输出img这幅图中每个目标的坐标，有几个输出几个
            save_annotations_and_imgs(coco, dataset, filename, objs)#路径不要用中文cv imread函数不支持中文路径


# 需要设置的路径
savepath = "W:/Jack_datasets/COCO/dataset/Jack_generate/COCO/"#路径保存处理后的文件
img_dir = savepath + 'images/' #此路径保存图像
anno_dir = savepath + 'annotations/' #此路径保存图像标注
datasets_list = ['train2017', 'val2017']

dataDir = 'W:/Jack_datasets/COCO/dataset/'#计划从此文件夹中的数据集中抽取数据
# coco有80类，这里写要提取类的名字，以person为例
classes_names = ['cat']

datasets_list = ['train2017', 'val2017']

#依次处理train和val的文件夹
# if __name__ == '__main__':
#     for dataset in datasets_list:
#         process_dataset(dataset, classes_names, dataDir, showimg, save_annotations_and_imgs)


#以上代码可以获取特定类别的数据图像，每个图像的json标注数据会提取出来，并为每个图像的标注转为xml格式（Pascovoc格式）(经验证正确的)



#下方代码可以和把多个xml格式的文件合并为一个json文件，方便mmdetection训练特定类别的分类器（合并的json文件有问题）


# -*- coding: utf-8 -*-
# @Time : 2019/8/27 10:48
# @Author :Rock
# @File : voc2coco.py
# just for object detection
import xml.etree.ElementTree as ET
import os
import json

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
image_set = set()

category_item_id = 0
image_id = 0
annotation_id = 0

def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id

def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    img_id = "%04d" % image_id
    image_id += 1
    image_item = dict()
    image_item['id'] = int(img_id)
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id

def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)

def parseXmlFiles(xml_path):
    for f in os.listdir(xml_path):
        if not f.endswith('.xml'):
            continue

        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        xml_file = os.path.join(xml_path, f)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('Pascal VOC XML root element should be annotation, rather than {}'.format(root.tag))

        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None

            if elem.tag == 'folder':
                continue

            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in image_set:
                    raise Exception('File name duplicated: {}'.format(file_name))

            elif current_image_id is None and file_name is not None and size['width'] is not None:
                current_image_id = addImgItem(file_name, size)

            for subelem in elem:
                bndbox['xmin'] = None
                bndbox['xmax'] = None
                bndbox['ymin'] = None
                bndbox['ymax'] = None

                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    if object_name not in category_set:
                        current_category_id = addCatItem(object_name)
                    else:
                        current_category_id = category_set[object_name]

                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('XML structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)

                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('XML structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)

                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('XML structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('XML structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('XML structure broken at bndbox tag')
                    bbox = [
                        bndbox['xmin'],
                        bndbox['ymin'],
                        bndbox['xmax'] - bndbox['xmin'],
                        bndbox['ymax'] - bndbox['ymin']
                    ]
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox)

if __name__ == '__main__':
    # xml_path = r'W:\Jack_datasets\COCO\dataset\Jack_generate_cat\COCO\annotations\\train2017\\'
    xml_path = r'W:\Jack_datasets\COCO\dataset\Jack_generate_cat\COCO\annotations\\val2017\\'
    #json_file = r'W:\Jack_datasets\COCO\dataset\Jack_generate_cat\COCO\annotations\instances_tran2017_cat.json'
    json_file = r'W:\Jack_datasets\COCO\dataset\Jack_generate_cat\COCO\annotations\instances_val2017_cat.json'
    parseXmlFiles(xml_path)
    with open(json_file, 'w') as json_fp:
        json.dump(coco, json_fp)

    # Debugging output
    print(f"Processed {len(coco['images'])} images.")
    print(f"Processed {len(coco['annotations'])} annotations.")
    print(f"Processed {len(coco['categories'])} categories.")

