from pathlib import Path


DATASET_ROOT = Path(
    r"E:\DataSets\MTSD_paper2\yolo54_paper2"
)

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def fix_label_names(split: str):
    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split

    if not image_dir.exists():
        raise FileNotFoundError(f"图片目录不存在：{image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"标签目录不存在：{label_dir}")

    image_files = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]

    label_files = [
        p for p in label_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    ]

    # 小写 stem -> 实际图片 stem
    image_map = {}

    for image_path in image_files:
        key = image_path.stem.lower()

        if key in image_map:
            raise RuntimeError(
                f"发现忽略大小写后重名图片："
                f"{image_map[key]} 和 {image_path.name}"
            )

        image_map[key] = image_path.stem

    renamed = 0
    already_correct = 0
    no_matching_image = []
    conflicts = []

    for label_path in label_files:
        key = label_path.stem.lower()

        if key not in image_map:
            no_matching_image.append(label_path.name)
            continue

        correct_stem = image_map[key]
        target_path = label_path.with_name(
            correct_stem + ".txt"
        )

        if label_path.name == target_path.name:
            already_correct += 1
            continue

        if target_path.exists():
            conflicts.append(
                f"{label_path.name} -> {target_path.name}"
            )
            continue

        # Windows 同一目录内仅修改大小写时，
        # 建议先改为临时名，再改为目标名
        temp_path = label_path.with_name(
            label_path.stem + "__TEMP_RENAME__.txt"
        )

        counter = 1
        while temp_path.exists():
            temp_path = label_path.with_name(
                f"{label_path.stem}__TEMP_RENAME__{counter}.txt"
            )
            counter += 1

        label_path.rename(temp_path)
        temp_path.rename(target_path)

        print(
            f"重命名：{label_path.name} -> {target_path.name}"
        )
        renamed += 1

    print(f"\n===== {split.upper()} 修复完成 =====")
    print(f"图片数：{len(image_files)}")
    print(f"标签数：{len(label_files)}")
    print(f"原本已经正确：{already_correct}")
    print(f"成功重命名：{renamed}")
    print(f"无匹配图片标签：{len(no_matching_image)}")
    print(f"目标文件冲突：{len(conflicts)}")

    if no_matching_image:
        print("\n前 20 个无匹配图片的标签：")
        for name in no_matching_image[:20]:
            print(name)

    if conflicts:
        print("\n前 20 个重命名冲突：")
        for item in conflicts[:20]:
            print(item)


if __name__ == "__main__":
    fix_label_names("train")
    fix_label_names("val")