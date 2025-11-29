import os
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pycocotools.coco import COCO


def apply_exif_orientation(image):
    """修正拍照图片方向（一般 MTSD 不会需要，但保留）"""
    _EXIF_ORIENT = 274
    if not hasattr(image, 'getexif'):
        return image
    try:
        exif = image.getexif()
        orientation = exif.get(_EXIF_ORIENT)
    except Exception:
        return image

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method:
        return image.transpose(method)
    return image


def draw_bboxes(ax, coco, anns):
    """在子图 ax 上画 bbox"""
    if len(anns) == 0:
        return

    # 每个类别一种颜色
    cat2color = {
        cat_id: (np.random.random((1, 3)) * 0.8 + 0.2).tolist()[0]
        for cat_id in coco.getCatIds()
    }

    polygons = []
    colors = []

    for ann in anns:
        color = cat2color[ann['category_id']]
        x, y, w, h = ann['bbox']  # COCO: [x, y, w, h]

        poly = Polygon(
            [(x, y), (x, y + h), (x + w, y + h), (x + w, y)],
            closed=True
        )
        polygons.append(poly)
        colors.append(color)

        cat_name = coco.loadCats(ann['category_id'])[0]['name']
        ax.text(
            x, y,
            cat_name,
            fontsize=6,
            color='white',
            bbox=dict(facecolor=color, alpha=0.7)
        )

    p = PatchCollection(
        polygons,
        facecolors=colors,
        edgecolors='white',
        linewidths=1,
        alpha=0.4
    )
    ax.add_collection(p)


def visualize_mtsd():
    # ============================
    # ⭐ 修改成你的 MTSD 路径
    # ============================
    ann_file = r"E:\DataSets\MTSD\yolo54\annotations\val.json"
    img_root = r"E:\DataSets\MTSD\yolo54\images"
    out_dir = r"E:\DataSets\MTSD\yolo54\images\val_see"

    # 创建输出文件夹
    os.makedirs(out_dir, exist_ok=True)

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    print(f"共 {len(img_ids)} 张图像，准备可视化写入 {out_dir}")

    for idx, img_id in enumerate(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']          # 可能是 'val/12.jpg'

        # 读图仍然按原始相对路径来拼
        img_path = osp.join(img_root, file_name)
        if not osp.exists(img_path):
            print(f"❌ 找不到图像：{img_path}")
            continue

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        # ======================
        # 画图
        # ======================
        image = Image.open(img_path).convert("RGB")
        image = apply_exif_orientation(image)

        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        draw_bboxes(plt.gca(), coco, anns)
        plt.axis('off')

        # ⭐ 保存时只保留文件名本身，全部扔到 val_see
        base_name = osp.basename(file_name)        # 例如 '12.jpg'
        save_path = osp.join(out_dir, base_name)   # E:\...\val_see\12.jpg

        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()

        if idx % 50 == 0:
            print(f"已处理 {idx}/{len(img_ids)} 张...")

    print(f"🎉 完成！所有可视化结果已保存到：{out_dir}")


if __name__ == "__main__":
    visualize_mtsd()
