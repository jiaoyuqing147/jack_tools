from pathlib import Path


# 只需要改这里的根目录。
ROOT_DIR = Path(r"E:\DataSets\MTSD")

# 原始数据
IMAGE_DIR = ROOT_DIR / "Detection"
GT_DETECTION = ROOT_DIR / "GT_Detection.txt"
GT_RECOGNITION = ROOT_DIR / "GT_Recognition.txt"
CLASSES66_TXT = ROOT_DIR / "classes66.txt"

# 统一输出目录
# yolo54 是旧结果；paper2 版本全部输出到 yolo54_paper2，避免混在一起。
OUTPUT_DIR = ROOT_DIR / "yolo54_paper2"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
REPORTS_DIR = OUTPUT_DIR / "reports"

# 中间文件
IMAGE_SIZES_CSV = REPORTS_DIR / "image_sizes.csv"
CLASS_COUNTS_CSV = REPORTS_DIR / "class_id_counts.csv"
LABELS_YOLO66_DIR = INTERMEDIATE_DIR / "labels_yolo66"
LABELS_PRUNED_FILTERED_DIR = INTERMEDIATE_DIR / "labels_pruned_filtered"

# 最终文件
LABELS_FINAL_DIR = OUTPUT_DIR / "labels"
CLASSES54_TXT = OUTPUT_DIR / "classes.txt"
NAMES54_YAML = OUTPUT_DIR / "names.yaml"

# 8:2 划分后的 YOLO 训练目录
SPLIT_IMAGES_TRAIN_DIR = OUTPUT_DIR / "images" / "train"
SPLIT_IMAGES_VAL_DIR = OUTPUT_DIR / "images" / "val"
SPLIT_LABELS_TRAIN_DIR = OUTPUT_DIR / "labels" / "train"
SPLIT_LABELS_VAL_DIR = OUTPUT_DIR / "labels" / "val"
SPLIT_TRAIN_TXT = OUTPUT_DIR / "train.txt"
SPLIT_VAL_TXT = OUTPUT_DIR / "val.txt"
DATA_YAML = OUTPUT_DIR / "data.yaml"

# 报告文件
MAPPING_CSV = REPORTS_DIR / "class_id_mapping.csv"
KEPT_IDS_TXT = REPORTS_DIR / "kept_class_ids_old_0based.txt"
REMOVED_IDS_TXT = REPORTS_DIR / "removed_class_ids_old_0based.txt"
REMOVED_CLASS_IDS_1BASED_TXT = REPORTS_DIR / "removed_class_ids_original_1based.txt"
MISSING_RECOGNITION_TXT = REPORTS_DIR / "missing_recognition.txt"
MISSING_IMAGE_SIZE_TXT = REPORTS_DIR / "missing_image_size.txt"
MISMATCH_CSV = REPORTS_DIR / "detection_recognition_mismatches.csv"
SUMMARY_TXT = REPORTS_DIR / "summary.txt"
SPLIT_IMAGE_LIST_CSV = REPORTS_DIR / "split_image_list.csv"
SPLIT_CLASS_STATS_CSV = REPORTS_DIR / "split_class_statistics.csv"
SPLIT_SUMMARY_TXT = REPORTS_DIR / "split_summary.txt"

# 删除出现次数 <= 3 的类别
DROP_MAX_COUNT = 3
ORIGINAL_CLASS_COUNT = 66


def make_output_dirs() -> None:
    for path in [
        OUTPUT_DIR,
        INTERMEDIATE_DIR,
        REPORTS_DIR,
        LABELS_YOLO66_DIR,
        LABELS_PRUNED_FILTERED_DIR,
        LABELS_FINAL_DIR,
        SPLIT_IMAGES_TRAIN_DIR,
        SPLIT_IMAGES_VAL_DIR,
        SPLIT_LABELS_TRAIN_DIR,
        SPLIT_LABELS_VAL_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
