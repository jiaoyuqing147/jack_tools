# -*- coding: utf-8 -*-
import os
import shutil
from glob import glob

# ===== 路径配置 =====
ROOT = r"E:\Datasets\tt100k_2021"

labels_train_dir = os.path.join(ROOT, "yolojack", "labels", "train")
labels_val_dir   = os.path.join(ROOT, "yolojack", "labels", "val")
labels_test_dir  = os.path.join(ROOT, "yolojack", "labels", "test")

# images_src_train = os.path.join(ROOT, "yolo", "images", "train")  # 供 train/val 使用
# images_src_test  = os.path.join(ROOT, "yolo", "images", "test")   # 供 test 使用

images_src_train = os.path.join(ROOT, "train")  # 供 train/val 使用
images_src_test  = os.path.join(ROOT, "test")   # 供 test 使用

images_dst_train = os.path.join(ROOT, "yolojack", "images", "train")
images_dst_val   = os.path.join(ROOT, "yolojack", "images", "val")
images_dst_test  = os.path.join(ROOT, "yolojack", "images", "test")

EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]
# ====================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def find_image(src_dir, stem):
    """在 src_dir 下按 stem 查找任意支持的扩展名图片，返回首个命中的完整路径，否则 None"""
    for ext in EXTS:
        cand = os.path.join(src_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    # 兜底：有些数据集用子文件夹，尝试通配搜索（稍慢）
    hits = []
    for ext in EXTS:
        hits += glob(os.path.join(src_dir, "**", stem + ext), recursive=True)
    return hits[0] if hits else None

def copy_by_labels(labels_dir, images_src_dir, images_dst_dir):
    """根据 labels_dir 下的 .txt，把同名图片从 images_src_dir 复制到 images_dst_dir"""
    ensure_dir(images_dst_dir)
    if not os.path.isdir(labels_dir):
        print(f"[WARN] labels 目录不存在：{labels_dir}")
        return dict(total_labels=0, found=0, copied=0, missing=0, missing_list=[])

    txts = [f for f in os.listdir(labels_dir) if f.lower().endswith(".txt")]
    copied = 0
    missing = []
    for i, txt in enumerate(sorted(txts)):
        stem = os.path.splitext(txt)[0]
        src_img = find_image(images_src_dir, stem)
        if not src_img:
            missing.append(stem)
            continue
        dst_img = os.path.join(images_dst_dir, os.path.basename(src_img))
        shutil.copy2(src_img, dst_img)
        copied += 1

        if (i + 1) % 500 == 0:
            print(f"  - 进度：{i+1}/{len(txts)}  已复制：{copied}  缺失：{len(missing)}")

    return dict(
        total_labels=len(txts),
        found=len(txts) - len(missing),
        copied=copied,
        missing=len(missing),
        missing_list=missing,
    )

def main():
    print("===== 复制 Train 图像（labels->images/train）=====")
    rep_train = copy_by_labels(labels_train_dir, images_src_train, images_dst_train)
    print(rep_train)

    print("\n===== 复制 Val 图像（labels->images/val）=====")
    rep_val = copy_by_labels(labels_val_dir, images_src_train, images_dst_val)
    print(rep_val)

    print("\n===== 复制 Test 图像（labels->images/test）=====")
    rep_test = copy_by_labels(labels_test_dir, images_src_test, images_dst_test)
    print(rep_test)

    # 汇总
    print("\n===== 汇总 =====")
    print(f"Train: 标签文件 {rep_train['total_labels']} -> 复制图片 {rep_train['copied']}，缺失 {rep_train['missing']}")
    print(f"Val  : 标签文件 {rep_val['total_labels']} -> 复制图片 {rep_val['copied']}，缺失 {rep_val['missing']}")
    print(f"Test : 标签文件 {rep_test['total_labels']} -> 复制图片 {rep_test['copied']}，缺失 {rep_test['missing']}")

    # 如需查看缺失清单，取消下面注释会打印前50个
    # if rep_train['missing'] or rep_val['missing'] or rep_test['missing']:
    #     print("\n前50个缺失样例（Train/Val/Test混合示例名）：")
    #     print((rep_train['missing_list'] + rep_val['missing_list'] + rep_test['missing_list'])[:50])

if __name__ == "__main__":
    main()
