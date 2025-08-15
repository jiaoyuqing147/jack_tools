import os
import glob
import shutil

# ===== 路径配置 =====
labels_root     = r"D:\Jiao\dataset\MTSD\MTSD\yolo54\labels"     # 已经分好 train/val/test 的 labels 目录
images_src_root = r"D:\Jiao\dataset\MTSD\MTSD\Detection"         # 原始图片根目录（场景图）
images_dst_root = r"D:\Jiao\dataset\MTSD\MTSD\yolo54\images"     # 目标 images 目录（将创建 train/val/test）

# 可能的图片扩展名（按顺序尝试）
valid_exts = [".jpg", ".jpeg", ".png", ".ppm"]

splits = ["train", "val", "test"]

# ===== 创建目标目录 =====
for sp in splits:
    os.makedirs(os.path.join(images_dst_root, sp), exist_ok=True)

def find_image(stem: str) -> str | None:
    """在 images_src_root 下尝试用多种扩展名找到同名图片，找到就返回完整路径，否则返回 None"""
    for ext in valid_exts:
        cand = os.path.join(images_src_root, stem + ext)
        if os.path.isfile(cand):
            return cand
    # 有些数据集文件名可能大小写不一致，保险起见再尝试小写文件名
    for ext in valid_exts:
        cand = os.path.join(images_src_root, stem.lower() + ext)
        if os.path.isfile(cand):
            return cand
    return None

def copy_images_for_split(split_name: str):
    labels_dir = os.path.join(labels_root, split_name)
    dst_dir    = os.path.join(images_dst_root, split_name)

    label_paths = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    if not label_paths:
        print(f"⚠️ {labels_dir} 下没有 .txt 标签文件")
        return

    copied, missing = 0, 0
    missing_list = []

    for lp in label_paths:
        stem = os.path.splitext(os.path.basename(lp))[0]  # 文件名不含扩展名
        src_img = find_image(stem)
        if src_img is None:
            missing += 1
            missing_list.append(stem)
            continue

        dst_img = os.path.join(dst_dir, os.path.basename(src_img))
        shutil.copy2(src_img, dst_img)
        copied += 1

    print(f"✅ [{split_name}] 已复制图片: {copied} 张 | 找不到对应图片: {missing} 张")
    if missing_list:
        # 只打印前20个，避免刷屏
        preview = ", ".join(missing_list[:20])
        more = "" if len(missing_list) <= 20 else f" ...（共{len(missing_list)}个）"
        print(f"   找不到的示例: {preview}{more}")

# ===== 执行 =====
for sp in splits:
    copy_images_for_split(sp)

print("\n🎉 全部完成：图片已按照 labels 的划分复制到 yolo54/images/train|val|test")
