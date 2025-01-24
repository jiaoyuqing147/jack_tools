#带上JSON标注的数据集可视化，为了发现问题，注意，可是话的图像编号是随机的
from fileinput import filename
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import numpy as np
import os.path as osp
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from PIL import Image


def apply_exif_orientation(image) :
    _EXIF_ORIENT = 274
    if not hasattr(image, 'getexif'):
        return image

    try:
        exif = image.getexif()
    except Exception:
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        image = image.transpose(method)
    return image

def show_bbox_only(coco, anns, show_label_bbox=True, is_filling=True):
    """Show bounding box of annotations only."""
    if len(anns) == 0:
        return

    ax = plt.gca()
    ax.set_autoscale_on(False)

    image2color = dict()
    for cat in coco.getCatIds():
        image2color[cat] = (np.random.random((1, 3)) * 0.7 + 0.3).tolist()[0]

    polygons = []
    colors = []

    for ann in anns:
        color = image2color[ann['category_id']]
        bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
        poly = Polygon([(bbox_x, bbox_y), (bbox_x, bbox_y + bbox_h),
                        (bbox_x + bbox_w, bbox_y + bbox_h), (bbox_x + bbox_w, bbox_y)],
                       closed=True)
        polygons.append(poly)
        colors.append(color)

        if show_label_bbox:
            label_bbox = dict(facecolor=color)
        else:
            label_bbox = None

        ax.text(
            bbox_x, bbox_y,
            '%s' % coco.loadCats(ann['category_id'])[0]['name'],
            color='white',
            bbox=label_bbox)

        # Filling the bounding boxes with color if enabled
        if is_filling:
            p = PatchCollection(polygons, facecolors=colors, linewidths=0, alpha=0.4)
            ax.add_collection(p)


# 初始化COCO工具
#coco = COCO('../cat_dataset/annotations/test.json')
#coco = COCO('W:/Jack_datasets/COCO/dataset/Jack_generate_cat/COCO/annotations/instances_tran2017_cat.json')
coco = COCO('W:/Jack_datasets/COCO/dataset/Jack_generate_cat/COCO/annotations/val2017_cat.json')
# 获取所有图片ID并打乱顺序
image_ids = coco.getImgIds()
np.random.shuffle(image_ids)

# 设置画布大小
plt.figure(figsize=(16, 5))

# 只可视化8张图片
for i in range(8):
    # 加载图片信息
    image_data = coco.loadImgs(image_ids[i])[0]
    # 获取图片路径
    #image_path = osp.join('../cat_dataset/images/', image_data['file_name'])
    #image_path = osp.join('W:/Jack_datasets/COCO/dataset/Jack_generate_cat/COCO/images/train2017/', image_data['file_name'])

    image_path = osp.join('W:/Jack_datasets/COCO/dataset/Jack_generate_cat/COCO/images/val2017',image_data['file_name'])
    # 获取该图片的所有注释ID
    annotation_ids = coco.getAnnIds(imgIds=image_data['id'], catIds=[], iscrowd=0)
    # 加载所有注释
    annotations = coco.loadAnns(annotation_ids)

    # 创建子图
    ax = plt.subplot(2, 4, i + 1)
    # 打开图片并转换为RGB格式
    image = Image.open(image_path).convert("RGB")

    # 根据EXIF数据调整图片方向（貌似使用opencv读取图像，好像就不需要考虑这些）
    #当使用手机或相机拍照时，设备的方向（横向或纵向）可能会影响到照片的最终方向。
    # 设备通常会通过内置传感器记录这一方向，并将其存储在照片的 EXIF 数据中。
    # 不同的设备和观看软件可能会以不同的方式解释这些信息。
    # 如果不根据这些EXIF数据调整图片方向，图片可能会在显示时被错误地旋转。
    image = apply_exif_orientation(image)

    # 显示图片
    ax.imshow(image)
    # 显示注释的边框
    show_bbox_only(coco, annotations)
    # 设置标题为图片文件名
    plt.title(f"{image_data['file_name']}")
    # 隐藏x和y轴的刻度
    plt.xticks([])
    plt.yticks([])

# 调整子图间距
plt.tight_layout()
# 显示最终结果
plt.show()