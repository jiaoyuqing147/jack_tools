import os, glob, shutil
from pathlib import Path
from PIL import Image

# ===== 配置 =====
src_images_dir = r"E:\DataSets\FullIJCNN2013"         # 原始图所在目录（ppm 在这里）
labels_root    = r"E:\DataSets\GTSDB\yolo43\labels"           # 按你的划分已有：labels/train|val|test
dst_images_root= r"E:\DataSets\GTSDB\yolo43\images"           # 目标：images/train|val|test
splits = ["train", "val", "test"]

# JPEG 存储质量
jpeg_quality = 95

# ===== 工具函数 =====
IMG_EXTS_SRC = (".ppm", ".jpg", ".jpeg", ".png")  # 在源里按这个顺序找
def find_src_image(stem: str):
    """按优先级在源目录寻找图片路径；优先 ppm"""
    for ext in IMG_EXTS_SRC:
        p = Path(src_images_dir) / f"{stem}{ext}"
        if p.exists():
            return str(p), ext.lower()
    return None, None

def save_as_jpg(src_path: str, dst_path: str):
    """把任意格式读入并保存为 JPG（RGB）"""
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        # 确保目标父目录存在
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        im.save(dst_path, format="JPEG", quality=jpeg_quality, optimize=True)

# ===== 主流程 =====
total_done = 0
missing = []

for sp in splits:
    lbl_dir = Path(labels_root) / sp
    out_dir = Path(dst_images_root) / sp
    out_dir.mkdir(parents=True, exist_ok=True)

    # 遍历该 split 下的所有 label txt
    for lp in glob.glob(str(lbl_dir / "*.txt")):
        stem = Path(lp).stem
        dst_img = out_dir / f"{stem}.jpg"
        if dst_img.exists():
            # 已存在就跳过（避免重复转换/拷贝）
            continue

        src_img, ext = find_src_image(stem)
        if not src_img:
            missing.append((sp, stem))
            continue

        ext = ext.lower()
        try:
            if ext == ".ppm":
                # ppm -> jpg
                save_as_jpg(src_img, str(dst_img))
            elif ext in (".jpg", ".jpeg"):
                # 直接拷贝为 jpg（保持后缀）
                shutil.copy2(src_img, dst_img)
            elif ext == ".png":
                # png -> jpg（有透明通道也会被转为RGB白底；如需纯白底可先填充）
                save_as_jpg(src_img, str(dst_img))
            else:
                # 兜底：统一转 jpg
                save_as_jpg(src_img, str(dst_img))
            total_done += 1
        except Exception as e:
            print(f"[ERROR] {sp}/{stem}: {e}")

print(f"✅ 完成转换/拷贝：{total_done} 张 -> {dst_images_root}")
if missing:
    print(f"⚠ 找不到源图的条目：{len(missing)}")
    # 如需查看具体缺失项，解开下行注释
    # for sp, stem in missing: print(sp, stem)
