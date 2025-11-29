import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import numpy as np
import os.path as osp
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from PIL import Image


def apply_exif_orientation(image):
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
    """只画 bbox + 类别名字"""
    if len(anns) == 0:
        return

    ax = plt.gca()
    ax.set_autoscale_on(False)

    # 给每个类别一个颜色
    cat2color = {
        cat_id: (np.random.random((1, 3)) * 0.7 + 0.3).tolist()[0]
        for cat_id in coco.getCatIds()
    }

    polygons = []
    colors = []

    for ann in anns:
        color = cat2color[ann['category_id']]
        x, y, w, h = ann['bbox']   # COCO 格式 [x, y, w, h]

        poly = Polygon(
            [(x, y),
             (x, y + h),
             (x + w, y + h),
             (x + w, y)],
            closed=True
        )
        polygons.append(poly)
        colors.append(color)

        if show_label_bbox:
            label_bbox = dict(facecolor=color, alpha=0.8)
        else:
            label_bbox = None

        # 类别名
        cat_name = coco.loadCats(ann['category_id'])[0]['name']
        ax.text(x, y,
                cat_name,
                color='white',
                fontsize=6,
                bbox=label_bbox)

    # 把所有 bbox 一次性画上去
    if is_filling and len(polygons) > 0:
        p = PatchCollection(polygons, facecolors=colors,
                            linewidths=1, edgecolors='w', alpha=0.4)
        ax.add_collection(p)


if __name__ == '__main__':
    # ========= 1. 改成你的 MTSD 路径 =========
    # Windows 示例：
    ann_file = r'E:\DataSets\MTSD\yolo54\annotations\val.json'   # 也可以用 train.json
    img_root = r'E:\DataSets\MTSD\yolo54\images'

    # Linux 示例（在服务器上跑就用这个）：
    # ann_file = r'/home/jiaoyuqing/bigspace/workspaceJack/datasets/MTSD/MTSD/yolo54/annotations/val.json'
    # img_root = r'/home/jiaoyuqing/bigspace/workspaceJack/datasets/MTSD/MTSD/yolo54/images'

    # ========= 2. 初始化 COCO =========
    coco = COCO(ann_file)

    img_ids = coco.getImgIds()
    np.random.shuffle(img_ids)

    plt.figure(figsize=(16, 8))

    # 随机看 8 张
    for i in range(8):
        img_info = coco.loadImgs(img_ids[i])[0]
        img_path = osp.join(img_root, img_info['file_name'])

        ann_ids = coco.getAnnIds(imgIds=img_info['id'], iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        ax = plt.subplot(2, 4, i + 1)
        image = Image.open(img_path).convert('RGB')
        image = apply_exif_orientation(image)

        ax.imshow(image)
        show_bbox_only(coco, anns)

        ax.set_title(img_info['file_name'], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
