# -*- coding: utf-8 -*-
from pathlib import Path
import shutil

# =========================
# 路径配置
# =========================
DATASET_ROOT = Path(r"E:\DataSets\tt100k_2021\yolojack")
IMG_DIR = DATASET_ROOT / "images" / "train"
LBL_DIR = DATASET_ROOT / "labels" / "train"

OUT_ROOT = Path(r"E:\DataSets\tt100k_2021\size_split_test")
OUT_TINY_IMG = OUT_ROOT / "tiny" / "images"
OUT_TINY_LBL = OUT_ROOT / "tiny" / "labels"
OUT_SMALL_IMG = OUT_ROOT / "small" / "images"
OUT_SMALL_LBL = OUT_ROOT / "small" / "labels"

IMG_SIZE = 640

# tiny: <16x16
TINY_TH = 16 * 16
# small: 16x16 ~ 32x32
SMALL_TH = 32 * 32


def ensure_dirs():
    for p in [OUT_TINY_IMG, OUT_TINY_LBL, OUT_SMALL_IMG, OUT_SMALL_LBL]:
        p.mkdir(parents=True, exist_ok=True)


def get_bucket(area_px: float):
    if area_px < TINY_TH:
        return "tiny"
    elif area_px < SMALL_TH:
        return "small"
    return None


def main():
    ensure_dirs()

    img_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    label_files = sorted(LBL_DIR.glob("*.txt"))

    tiny_img_count = 0
    small_img_count = 0

    for lbl_file in label_files:
        has_tiny = False
        has_small = False

        lines = lbl_file.read_text(encoding="utf-8").strip().splitlines() if lbl_file.exists() else []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            _, xc, yc, w, h = map(float, parts)
            w_px = w * IMG_SIZE
            h_px = h * IMG_SIZE
            area = w_px * h_px

            bucket = get_bucket(area)
            if bucket == "tiny":
                has_tiny = True
            elif bucket == "small":
                has_small = True

        # 找对应图片
        img_file = None
        for ext in img_exts:
            candidate = IMG_DIR / f"{lbl_file.stem}{ext}"
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            print(f"[警告] 找不到对应图片: {lbl_file.stem}")
            continue

        if has_tiny:
            shutil.copy2(img_file, OUT_TINY_IMG / img_file.name)
            shutil.copy2(lbl_file, OUT_TINY_LBL / lbl_file.name)
            tiny_img_count += 1

        if has_small:
            shutil.copy2(img_file, OUT_SMALL_IMG / img_file.name)
            shutil.copy2(lbl_file, OUT_SMALL_LBL / lbl_file.name)
            small_img_count += 1

    print("=" * 60)
    print("完成")
    print(f"Tiny 图片数 : {tiny_img_count}")
    print(f"Small 图片数: {small_img_count}")
    print(f"输出目录: {OUT_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()