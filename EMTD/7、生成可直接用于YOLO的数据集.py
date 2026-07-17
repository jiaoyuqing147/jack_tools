import shutil
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


# ============================================================
# 1. 路径配置：按你的实际目录修改
# ============================================================

# 原始图片目录
SOURCE_IMAGE_DIR = Path(
    r"E:\DataSets\EMTD\Detection"
)

# 第3步生成的 YOLO 标签目录
SOURCE_LABEL_DIR = Path(
    r"E:\UKMJIAO\AlgorithmCodes\jack_tools\EMTD\labels_yolo_66classes"
)

# 第6步生成的划分列表目录
SPLIT_DIR = Path(
    r"E:\UKMJIAO\AlgorithmCodes\jack_tools\EMTD\optimized_split_80_20_fast"
)

TRAIN_LIST_PATH = SPLIT_DIR / "train_images.csv"
VAL_LIST_PATH = SPLIT_DIR / "test_images.csv"

# 第1步生成的 classes.txt
CLASSES_FILE = Path(
    r"E:\UKMJIAO\AlgorithmCodes\jack_tools\EMTD\classes.txt"
)

# 最终输出目录
YOLO_DATASET_DIR = Path(
    r"E:\DataSets\EMTD\EMTD_YOLO_66"
)

TRAIN_IMAGE_DIR = YOLO_DATASET_DIR / "images" / "train"
VAL_IMAGE_DIR = YOLO_DATASET_DIR / "images" / "val"
TRAIN_LABEL_DIR = YOLO_DATASET_DIR / "labels" / "train"
VAL_LABEL_DIR = YOLO_DATASET_DIR / "labels" / "val"
DATA_YAML_PATH = YOLO_DATASET_DIR / "EMTD_66.yaml"

# True：运行前清空旧的 train/val 文件，避免上次残留
CLEAR_DESTINATION = True

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


# ============================================================
# 2. 工具函数
# ============================================================

def normalize_filename(value: str) -> str:
    """统一为小写文件名，不保留目录。"""
    return Path(str(value).strip()).name.lower()


def prepare_directory(path: Path) -> None:
    if CLEAR_DESTINATION and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_split_list(csv_path: Path) -> List[str]:
    """读取第6步生成的无表头单列 CSV。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到划分文件：{csv_path}")

    df = pd.read_csv(csv_path, header=None, dtype=str, keep_default_na=False)

    names = [
        normalize_filename(value)
        for value in df.iloc[:, 0].tolist()
        if str(value).strip()
    ]

    # 保持原顺序去重
    unique_names = list(dict.fromkeys(names))

    if len(unique_names) != len(names):
        print(
            f"⚠️ {csv_path.name} 中有 {len(names) - len(unique_names)} 个重复项，"
            "已自动去重。"
        )

    return unique_names


def build_file_index(
    directory: Path,
    allowed_extensions: Set[str],
) -> Dict[str, Path]:
    """建立大小写不敏感的文件名索引。"""
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在：{directory}")

    index: Dict[str, Path] = {}

    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_extensions:
            continue

        key = path.name.lower()

        if key in index:
            raise RuntimeError(
                "源目录中存在仅大小写不同或完全同名的文件：\n"
                f"{index[key]}\n{path}"
            )

        index[key] = path

    return index


def load_class_names(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 classes.txt：{path}")

    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(names) != 66:
        raise ValueError(f"classes.txt 应为66类，当前是 {len(names)} 类。")

    return names


def validate_label(label_path: Path, num_classes: int) -> tuple[int, List[str]]:
    """检查 YOLO 标签格式及类别编号、归一化坐标。"""
    errors: List[str] = []
    text = label_path.read_text(encoding="utf-8").strip()

    if not text:
        return 0, [f"空标签：{label_path}"]

    box_count = 0

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()

        if len(parts) != 5:
            errors.append(
                f"{label_path.name} 第{line_no}行字段数不是5：{line}"
            )
            continue

        try:
            class_id = int(parts[0])
            x, y, w, h = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(
                f"{label_path.name} 第{line_no}行存在非数值内容：{line}"
            )
            continue

        if not 0 <= class_id < num_classes:
            errors.append(
                f"{label_path.name} 第{line_no}行类别越界：{class_id}"
            )

        if not all(0.0 <= value <= 1.0 for value in [x, y, w, h]):
            errors.append(
                f"{label_path.name} 第{line_no}行坐标未归一化：{line}"
            )

        if w <= 0 or h <= 0:
            errors.append(
                f"{label_path.name} 第{line_no}行宽高不合法：{line}"
            )

        if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6:
            errors.append(
                f"{label_path.name} 第{line_no}行水平框越界：{line}"
            )

        if y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
            errors.append(
                f"{label_path.name} 第{line_no}行垂直框越界：{line}"
            )

        box_count += 1

    return box_count, errors


def copy_split(
    split_name: str,
    filenames: List[str],
    image_dest: Path,
    label_dest: Path,
    image_index: Dict[str, Path],
    label_index: Dict[str, Path],
    num_classes: int,
) -> dict:
    statistics = {
        "images": 0,
        "labels": 0,
        "instances": 0,
        "errors": [],
    }

    for i, filename in enumerate(filenames, start=1):
        source_image = image_index.get(filename)

        if source_image is None:
            statistics["errors"].append(f"缺失图片：{filename}")
            continue

        label_name = f"{Path(filename).stem}.txt"
        source_label = label_index.get(label_name.lower())

        if source_label is None:
            statistics["errors"].append(f"缺失标签：{label_name}")
            continue

        box_count, label_errors = validate_label(source_label, num_classes)

        if label_errors:
            statistics["errors"].extend(label_errors)
            continue

        shutil.copy2(source_image, image_dest / source_image.name)
        shutil.copy2(source_label, label_dest / label_name)

        statistics["images"] += 1
        statistics["labels"] += 1
        statistics["instances"] += box_count

        if i % 200 == 0 or i == len(filenames):
            print(
                f"  {split_name}: {i}/{len(filenames)} "
                f"({i / len(filenames):.1%})"
            )

    return statistics


def yaml_quote(value: str) -> str:
    """为 YAML 字符串进行简单安全引用。"""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_data_yaml(class_names: List[str]) -> None:
    root = str(YOLO_DATASET_DIR.resolve()).replace("\\", "/")

    lines = [
        f"path: {yaml_quote(root)}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
    ]

    for index, name in enumerate(class_names):
        lines.append(f"  {index}: {yaml_quote(name)}")

    DATA_YAML_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# 3. 主程序
# ============================================================

def main() -> None:
    print("===== 生成可直接用于 YOLO 的 EMTD 数据集 =====")

    class_names = load_class_names(CLASSES_FILE)

    train_names = read_split_list(TRAIN_LIST_PATH)
    val_names = read_split_list(VAL_LIST_PATH)

    overlap = set(train_names) & set(val_names)
    if overlap:
        raise RuntimeError(
            f"训练集和验证集有 {len(overlap)} 张重复图片："
            f"{sorted(overlap)[:20]}"
        )

    print(f"训练集列表：{len(train_names)} 张")
    print(f"验证集列表：{len(val_names)} 张")
    print("训练集与验证集无重复。")

    for directory in [
        TRAIN_IMAGE_DIR,
        VAL_IMAGE_DIR,
        TRAIN_LABEL_DIR,
        VAL_LABEL_DIR,
    ]:
        prepare_directory(directory)

    image_index = build_file_index(SOURCE_IMAGE_DIR, IMAGE_EXTENSIONS)
    label_index = build_file_index(SOURCE_LABEL_DIR, {".txt"})

    print(f"源图片数：{len(image_index)}")
    print(f"源标签数：{len(label_index)}")

    print("\n正在整理训练集……")
    train_stats = copy_split(
        "train",
        train_names,
        TRAIN_IMAGE_DIR,
        TRAIN_LABEL_DIR,
        image_index,
        label_index,
        len(class_names),
    )

    print("\n正在整理验证集……")
    val_stats = copy_split(
        "val",
        val_names,
        VAL_IMAGE_DIR,
        VAL_LABEL_DIR,
        image_index,
        label_index,
        len(class_names),
    )

    all_errors = train_stats["errors"] + val_stats["errors"]

    if all_errors:
        error_path = YOLO_DATASET_DIR / "dataset_build_errors.txt"
        error_path.write_text("\n".join(all_errors), encoding="utf-8")
        raise RuntimeError(
            f"发现 {len(all_errors)} 个问题，详情见：{error_path}"
        )

    if train_stats["images"] != len(train_names):
        raise RuntimeError("训练集实际复制数量与划分列表不一致。")

    if val_stats["images"] != len(val_names):
        raise RuntimeError("验证集实际复制数量与划分列表不一致。")

    write_data_yaml(class_names)

    summary = pd.DataFrame(
        [
            {
                "split": "train",
                "images": train_stats["images"],
                "labels": train_stats["labels"],
                "instances": train_stats["instances"],
            },
            {
                "split": "val",
                "images": val_stats["images"],
                "labels": val_stats["labels"],
                "instances": val_stats["instances"],
            },
        ]
    )

    summary.to_csv(
        YOLO_DATASET_DIR / "dataset_build_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== 构建完成 =====")
    print(f"训练图片：{train_stats['images']}")
    print(f"训练标签：{train_stats['labels']}")
    print(f"训练实例：{train_stats['instances']}")
    print(f"验证图片：{val_stats['images']}")
    print(f"验证标签：{val_stats['labels']}")
    print(f"验证实例：{val_stats['instances']}")
    print(f"数据集目录：{YOLO_DATASET_DIR.resolve()}")
    print(f"YAML：{DATA_YAML_PATH.resolve()}")


if __name__ == "__main__":
    main()
