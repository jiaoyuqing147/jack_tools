from mtsd_case_utils import IMAGE_SUFFIXES, build_unique_stem_index, validate_yolo_file
from mtsd_paths_config import (
    SPLIT_IMAGES_TRAIN_DIR, SPLIT_IMAGES_VAL_DIR, SPLIT_LABELS_TRAIN_DIR, SPLIT_LABELS_VAL_DIR,
)

K = 54


def validate_split(name, image_dir, label_dir):
    images = build_unique_stem_index(image_dir, IMAGE_SUFFIXES)
    labels = build_unique_stem_index(label_dir, {".txt"})
    for label in labels.values():
        validate_yolo_file(label, K, allow_empty=False)
    exact_images = {path.stem for path in images.values()}
    exact_labels = {path.stem for path in labels.values()}
    folded_images, folded_labels = set(images), set(labels)
    image_only = exact_images - exact_labels
    label_only = exact_labels - exact_images
    if folded_images != folded_labels:
        raise RuntimeError(f"{name}: missing counterpart ignoring case")
    if image_only or label_only:
        raise RuntimeError(f"{name}: case mismatch: image-only={sorted(image_only)[:10]}, label-only={sorted(label_only)[:10]}")
    print(name)
    print(f"图片数：{len(images)}")
    print(f"标签数：{len(labels)}")
    print(f"严格同名匹配：{len(exact_images & exact_labels)}")
    print(f"有图片但无同名标签：{len(image_only)}")
    print(f"有标签但无同名图片：{len(label_only)}")
    return folded_images


def main():
    train = validate_split("TRAIN", SPLIT_IMAGES_TRAIN_DIR, SPLIT_LABELS_TRAIN_DIR)
    val = validate_split("VAL", SPLIT_IMAGES_VAL_DIR, SPLIT_LABELS_VAL_DIR)
    overlap = train & val
    if overlap:
        raise RuntimeError(f"Train/val duplicate stems ignoring case: {sorted(overlap)[:10]}")
    print("train/val 重复：0")
    print("最终数据集验证通过")


if __name__ == "__main__":
    main()
