import os
from pathlib import Path
import cv2

from aug_albu_custom import AugConfig, build_train_augmentations_yolo


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_yolo_txt(txt_path: Path):
    """
    读取 YOLO txt：
    每行：cls xc yc w h （归一化）
    返回：bboxes(list[tuple]), labels(list[int])
    """
    bboxes, labels = [], []
    if not txt_path.exists():
        return bboxes, labels

    for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        # 简单合法性裁剪（防止少数脏标注）
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        bboxes.append((xc, yc, w, h))
        labels.append(cls)

    return bboxes, labels


def write_yolo_txt(txt_path: Path, bboxes, labels):
    """
    写回 YOLO txt：cls xc yc w h
    """
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for (xc, yc, w, h), cls in zip(bboxes, labels):
        lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def augment_split(
    images_dir: Path,
    labels_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    repeats: int = 1,
):
    """
    repeats=1 表示每张图生成 1 张增强图
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    # 你可以在这里调增强强度/概率
    cfg = AugConfig(
        enable_resize=False,      # 一般不建议在这里强行 Resize，留给 YOLO 的 letterbox
        out_size=640,             # enable_resize=True 才会生效
        p_weather=0.20,
        p_color=0.35,
        p_blur=0.25,
        p_distort=0.15,
        p_dropout=0.15,
        # 小目标建议 CoarseDropout 不要太大
        max_holes=6,
        max_height=24,
        max_width=24,
    )

    aug = build_train_augmentations_yolo(cfg, min_visibility=0.20)

    img_paths = []
    for p in images_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            img_paths.append(p)

    print(f"Found {len(img_paths)} images under: {images_dir}")

    for img_path in img_paths:
        rel = img_path.relative_to(images_dir)
        label_path = labels_dir / rel.with_suffix(".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] failed to read image: {img_path}")
            continue

        bboxes, labels = read_yolo_txt(label_path)

        # 如果某些图没有标注，也可以照样增强并输出空txt（可选）
        for k in range(repeats):
            out = aug(image=img, bboxes=bboxes, class_labels=labels)
            img_aug = out["image"]
            bboxes_aug = out["bboxes"]
            labels_aug = out["class_labels"]

            # 输出文件名：原名 + _aug{idx}
            stem = img_path.stem
            out_rel = rel.with_name(f"{stem}_aug{k}").with_suffix(img_path.suffix)

            out_img_path = out_images_dir / out_rel
            out_txt_path = out_labels_dir / out_rel.with_suffix(".txt")

            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img_path), img_aug)

            write_yolo_txt(out_txt_path, bboxes_aug, labels_aug)

    print("Done.")


if __name__ == "__main__":
    # 改成你的实际路径
    dataset_root = Path(r"E:\DataSets\your_dataset")  # Windows 举例

    images_dir = dataset_root / "images" / "train"
    labels_dir = dataset_root / "labels" / "train"

    out_images_dir = dataset_root / "images_aug" / "train"
    out_labels_dir = dataset_root / "labels_aug" / "train"

    augment_split(
        images_dir=images_dir,
        labels_dir=labels_dir,
        out_images_dir=out_images_dir,
        out_labels_dir=out_labels_dir,
        repeats=2,   # 每张图生成 2 张增强图
    )
