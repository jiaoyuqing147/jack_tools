from pathlib import Path


DATASET_ROOT = Path(r"E:\DataSets\MTSD\yolo54_paper2")

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"
}


def check_split(split: str):
    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split

    print(f"\n{'=' * 20} {split.upper()} {'=' * 20}")
    print(f"图片目录：{image_dir}")
    print(f"标签目录：{label_dir}")

    if not image_dir.exists():
        print(f"❌ 图片目录不存在：{image_dir}")
        return

    if not label_dir.exists():
        print(f"❌ 标签目录不存在：{label_dir}")
        return

    image_files = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }
    ]

    label_files = list(label_dir.rglob("*.txt"))

    image_stems = {p.stem for p in image_files}
    label_stems = {p.stem for p in label_files}

    matched = image_stems & label_stems
    images_without_labels = image_stems - label_stems
    labels_without_images = label_stems - image_stems

    print(f"图片数：{len(image_files)}")
    print(f"标签数：{len(label_files)}")
    print(f"严格同名匹配：{len(matched)}")
    print(f"有图片但无同名标签：{len(images_without_labels)}")
    print(f"有标签但无同名图片：{len(labels_without_images)}")

    print("\n前 20 个有图片但无同名标签：")
    for name in sorted(images_without_labels)[:20]:
        print(f"  {name}")

    print("\n前 20 个有标签但无同名图片：")
    for name in sorted(labels_without_images)[:20]:
        print(f"  {name}")

    # 忽略大小写后检查
    image_stems_lower = {p.stem.lower() for p in image_files}
    label_stems_lower = {p.stem.lower() for p in label_files}
    matched_lower = image_stems_lower & label_stems_lower

    print(f"\n忽略大小写后匹配：{len(matched_lower)}")

    if len(matched_lower) > len(matched):
        print("⚠️ 存在文件名大小写不一致问题")

    # 检查可能多了一层图片扩展名
    suspicious_labels = []

    for p in label_files:
        stem_lower = p.stem.lower()

        if stem_lower.endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            suspicious_labels.append(p.name)

    print(f"标签名中疑似多带图片扩展名：{len(suspicious_labels)}")

    for name in suspicious_labels[:20]:
        print(f"  {name}")


if __name__ == "__main__":
    check_split("train")
    check_split("val")