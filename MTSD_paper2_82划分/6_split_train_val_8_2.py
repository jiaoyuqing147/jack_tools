import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from mtsd_paths_config import (
    LABELS_FINAL_DIR,
    OUTPUT_DIR,
    REPORTS_DIR,
    SPLIT_CLASS_STATS_CSV,
    SPLIT_IMAGE_LIST_CSV,
    SPLIT_LABELS_TRAIN_DIR,
    SPLIT_LABELS_VAL_DIR,
    SPLIT_SUMMARY_TXT,
    make_output_dirs,
)


K = 54
VAL_RATIO = 0.20
KEEP_EMPTY = False
MAX_TRIALS = 300
BASE_SEED = 42


def read_label_classes(label_path: Path) -> list[int]:
    classes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            class_id = int(line.split()[0])
        except ValueError:
            continue
        if 0 <= class_id < K:
            classes.append(class_id)
    return classes


def clear_label_split_dirs() -> None:
    for split_dir in [SPLIT_LABELS_TRAIN_DIR, SPLIT_LABELS_VAL_DIR]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def build_multilabel_matrix() -> tuple[list[Path], np.ndarray, list[list[int]], int]:
    label_paths = sorted(LABELS_FINAL_DIR.glob("*.txt"))
    used_paths = []
    label_classes = []
    matrix_rows = []
    dropped_empty = 0

    for label_path in label_paths:
        classes = read_label_classes(label_path)
        if not classes and not KEEP_EMPTY:
            dropped_empty += 1
            continue

        y = np.zeros(K, dtype=int)
        for class_id in set(classes):
            y[class_id] = 1

        used_paths.append(label_path)
        label_classes.append(classes)
        matrix_rows.append(y)

    if not used_paths:
        raise RuntimeError(f"No usable label files found in: {LABELS_FINAL_DIR}")

    return used_paths, np.array(matrix_rows, dtype=int), label_classes, dropped_empty


def split_once(y_matrix: np.ndarray, seed: int) -> tuple[list[int], list[int]]:
    n = len(y_matrix)
    target_val_count = int(round(n * VAL_RATIO))
    target_val_count = max(1, min(target_val_count, n - 1))

    n_splits = max(2, int(round(1 / VAL_RATIO)))
    splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    x_dummy = np.zeros((n, 1))
    best_train = None
    best_val = None
    best_gap = 10**9

    for train_idx, val_idx in splitter.split(x_dummy, y_matrix):
        gap = abs(len(val_idx) - target_val_count)
        if gap < best_gap:
            best_gap = gap
            best_train = train_idx.tolist()
            best_val = val_idx.tolist()

    return best_train, best_val


def box_counter(label_classes: list[list[int]], idxs: list[int]) -> Counter:
    counter = Counter()
    for idx in idxs:
        counter.update(label_classes[idx])
    return counter


def score_split(label_classes: list[list[int]], train_idx: list[int], val_idx: list[int]) -> tuple:
    train_counter = box_counter(label_classes, train_idx)
    val_counter = box_counter(label_classes, val_idx)
    total_counter = train_counter + val_counter

    train_cov = sum(train_counter[c] > 0 for c in range(K))
    val_cov = sum(val_counter[c] > 0 for c in range(K))

    total_boxes = sum(total_counter.values())
    val_boxes = sum(val_counter.values())
    val_box_ratio = val_boxes / total_boxes if total_boxes else 0
    ratio_gap = abs(val_box_ratio - VAL_RATIO)

    min_train = min(train_counter[c] for c in range(K) if total_counter[c] > 0)
    min_val = min(val_counter[c] for c in range(K) if total_counter[c] > 0)

    # Larger tuple is better. Prefer full train/val coverage first, then avoid
    # starving train, then keep box ratio close to 8:2.
    return (
        train_cov,
        val_cov,
        min_train,
        min_val,
        -ratio_gap,
        -abs(len(val_idx) - round((len(train_idx) + len(val_idx)) * VAL_RATIO)),
    )


def find_best_split(y_matrix: np.ndarray, label_classes: list[list[int]]) -> tuple[list[int], list[int], int, tuple]:
    best_result = None
    best_score = None

    for trial in range(MAX_TRIALS):
        seed = BASE_SEED + trial
        train_idx, val_idx = split_once(y_matrix, seed)
        score = score_split(label_classes, train_idx, val_idx)

        if best_score is None or score > best_score:
            best_score = score
            best_result = (train_idx, val_idx, seed)

        if score[0] == K and score[1] == K and score[2] >= 2 and score[3] >= 1:
            break

    train_idx, val_idx, seed = best_result
    return train_idx, val_idx, seed, best_score


def copy_labels(label_paths: list[Path], idxs: list[int], dst_dir: Path) -> None:
    for idx in idxs:
        shutil.copy2(label_paths[idx], dst_dir / label_paths[idx].name)


def write_reports(label_paths: list[Path], label_classes: list[list[int]], train_idx: list[int], val_idx: list[int], used_seed: int, dropped_empty: int) -> None:
    train_counter = box_counter(label_classes, train_idx)
    val_counter = box_counter(label_classes, val_idx)
    total_counter = train_counter + val_counter

    split_rows = []
    for split_name, idxs in [("train", train_idx), ("val", val_idx)]:
        for idx in idxs:
            split_rows.append({
                "filename": label_paths[idx].name,
                "split": split_name,
                "boxes": len(label_classes[idx]),
                "classes": " ".join(map(str, sorted(set(label_classes[idx])))),
            })
    pd.DataFrame(split_rows).to_csv(SPLIT_IMAGE_LIST_CSV, index=False)

    class_rows = []
    for class_id in range(K):
        class_rows.append({
            "class_id": class_id,
            "train_count": train_counter[class_id],
            "val_count": val_counter[class_id],
            "total_count": total_counter[class_id],
        })
    pd.DataFrame(class_rows).to_csv(SPLIT_CLASS_STATS_CSV, index=False)

    train_boxes = sum(train_counter.values())
    val_boxes = sum(val_counter.values())
    total_boxes = train_boxes + val_boxes
    val_box_ratio = val_boxes / total_boxes if total_boxes else 0

    summary_lines = [
        "MTSD YOLO54 paper2 label-only multilabel 8:2 split summary",
        f"method: MultilabelStratifiedKFold",
        f"seed: {used_seed}",
        f"val_ratio_target: {VAL_RATIO}",
        f"source_labels: {LABELS_FINAL_DIR}",
        f"output_root: {OUTPUT_DIR}",
        f"usable_label_files: {len(label_paths)}",
        f"dropped_empty_label_files: {dropped_empty}",
        f"train_label_files: {len(train_idx)}",
        f"val_label_files: {len(val_idx)}",
        f"train_boxes: {train_boxes}",
        f"val_boxes: {val_boxes}",
        f"val_box_ratio: {val_box_ratio:.4f}",
        f"train_class_coverage: {sum(train_counter[c] > 0 for c in range(K))}/{K}",
        f"val_class_coverage: {sum(val_counter[c] > 0 for c in range(K))}/{K}",
        f"split_class_stats: {SPLIT_CLASS_STATS_CSV}",
        f"split_file_list: {SPLIT_IMAGE_LIST_CSV}",
    ]
    SPLIT_SUMMARY_TXT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


def main() -> None:
    make_output_dirs()
    clear_label_split_dirs()
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)

    label_paths, y_matrix, label_classes, dropped_empty = build_multilabel_matrix()
    train_idx, val_idx, used_seed, best_score = find_best_split(y_matrix, label_classes)

    copy_labels(label_paths, train_idx, SPLIT_LABELS_TRAIN_DIR)
    copy_labels(label_paths, val_idx, SPLIT_LABELS_VAL_DIR)
    write_reports(label_paths, label_classes, train_idx, val_idx, used_seed, dropped_empty)

    print("Best score:", best_score)
    print(f"Train labels: {SPLIT_LABELS_TRAIN_DIR}")
    print(f"Val labels: {SPLIT_LABELS_VAL_DIR}")


if __name__ == "__main__":
    main()