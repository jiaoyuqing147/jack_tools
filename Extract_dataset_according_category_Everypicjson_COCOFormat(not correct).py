import os
import shutil
import json
from pycocotools.coco import COCO

#不知为何，函数提取的图像的个数不全，数量很少，跑不完所有的图像
def extract_category_images(dataDir, datafile, class_id, output_folder):
    """
    提取指定类别的图像和标注并保存到指定文件夹。
    在指定文件夹存放所有指定类别的图像，每个图像有个同名的json文件（标注信息）

    :param dataDir: COCO数据集的路径
    :param datafile: 文件夹名称（如'val2017'）
    :param class_id: 要提取的类别ID
    :param output_folder: 存放提取结果的文件夹
    """
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, datafile)
    coco = COCO(annFile)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    imgIds = coco.getImgIds(catIds=[class_id])

    print(f"Found {len(imgIds)} images with class_id {class_id}")

    processed_images = 0
    for imgId in imgIds:
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=[class_id])
        anns = coco.loadAnns(annIds)

        if len(anns) == 0:
            continue  # Skip images with no annotations for the specified class

        img_src_path = os.path.join(dataDir, datafile, img['file_name'])
        img_dst_path = os.path.join(output_folder, img['file_name'])

        try:
            shutil.copy(img_src_path, img_dst_path)
            json.dump(anns, open(os.path.join(output_folder, img['file_name'].split('.')[0] + '.json'), 'w'))
            processed_images += 1
        except Exception as e:
            print(f"Error processing image {img['file_name']}: {e}")

    print(f"Processed {processed_images} images with class_id {class_id}")


# 使用示例
dataDir = 'W:/Jack_datasets/COCO/dataset'
datafile = 'val2017'
class_id = 17  # 'cat' 类别的ID
output_folder = 'W:/Jack_datasets/COCO/dataset/output_folder'
extract_category_images(dataDir, datafile, class_id, output_folder)
