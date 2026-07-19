import random
import shutil
from collections import Counter

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from mtsd_case_utils import IMAGE_SUFFIXES, build_unique_stem_index, recreate_owned_directory, validate_yolo_file
from mtsd_paths_config import (
    IMAGE_DIR, LABELS_FINAL_DIR, OUTPUT_DIR, SPLIT_CLASS_STATS_CSV, SPLIT_IMAGE_LIST_CSV,
    SPLIT_LABELS_TRAIN_DIR, SPLIT_LABELS_VAL_DIR, SPLIT_SUMMARY_TXT, make_output_dirs,
)

K = 54
VAL_RATIO = 0.20
KEEP_EMPTY = False
MAX_TRIALS = 300
BASE_SEED = 42


def load_labels():
    images = build_unique_stem_index(IMAGE_DIR, IMAGE_SUFFIXES)
    labels = build_unique_stem_index(LABELS_FINAL_DIR, {".txt"})
    paths, names, classes_per_file, rows = [], [], [], []
    dropped_empty = 0
    # Preserve the original script's exact Path sorting/order because row order
    # participates in the deterministic seeded split.
    for label in sorted(labels.values()):
        key = label.stem.casefold()
        image = images.get(key)
        if image is None:
            raise FileNotFoundError(f"Final label has no source image: {label}")
        validate_yolo_file(label, K)
        classes = [int(line.split()[0]) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not classes and not KEEP_EMPTY:
            dropped_empty += 1
            continue
        row = np.zeros(K, dtype=int)
        for class_id in set(classes):
            row[class_id] = 1
        paths.append(label)
        names.append(f"{image.stem}.txt")  # real source-image stem
        classes_per_file.append(classes)
        rows.append(row)
    if not paths:
        raise RuntimeError(f"No usable labels in {LABELS_FINAL_DIR}")
    return paths, names, np.asarray(rows), classes_per_file, dropped_empty


def split_once(matrix, seed):
    count = len(matrix)
    target = max(1, min(int(round(count * VAL_RATIO)), count - 1))
    splitter = MultilabelStratifiedKFold(n_splits=max(2, int(round(1 / VAL_RATIO))), shuffle=True, random_state=seed)
    best = None
    best_gap = 10**9
    for train, val in splitter.split(np.zeros((count, 1)), matrix):
        gap = abs(len(val) - target)
        if gap < best_gap:
            best_gap, best = gap, (train.tolist(), val.tolist())
    return best


def counts(classes_per_file, indices):
    result = Counter()
    for index in indices:
        result.update(classes_per_file[index])
    return result


def score(classes_per_file, train, val):
    train_counts, val_counts = counts(classes_per_file, train), counts(classes_per_file, val)
    total = train_counts + val_counts
    total_boxes = sum(total.values())
    val_ratio = sum(val_counts.values()) / total_boxes if total_boxes else 0
    return (
        sum(train_counts[c] > 0 for c in range(K)), sum(val_counts[c] > 0 for c in range(K)),
        min(train_counts[c] for c in range(K) if total[c] > 0),
        min(val_counts[c] for c in range(K) if total[c] > 0),
        -abs(val_ratio - VAL_RATIO), -abs(len(val) - round((len(train) + len(val)) * VAL_RATIO)),
    )


def choose_split(matrix, classes_per_file):
    best = None
    best_score = None
    for trial in range(MAX_TRIALS):
        seed = BASE_SEED + trial
        train, val = split_once(matrix, seed)
        candidate = score(classes_per_file, train, val)
        if best_score is None or candidate > best_score:
            best_score, best = candidate, (train, val, seed)
        if candidate[0] == K and candidate[1] == K and candidate[2] >= 2 and candidate[3] >= 1:
            break
    return *best, best_score


def main() -> None:
    make_output_dirs()
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    paths, names, matrix, classes_per_file, dropped_empty = load_labels()
    train, val, seed, best_score = choose_split(matrix, classes_per_file)
    if set(train) & set(val) or set(train) | set(val) != set(range(len(paths))):
        raise RuntimeError("Invalid train/val partition")

    # Preflight finished: now clean exactly the two directories owned by this step.
    recreate_owned_directory(SPLIT_LABELS_TRAIN_DIR)
    recreate_owned_directory(SPLIT_LABELS_VAL_DIR)
    for indices, destination in ((train, SPLIT_LABELS_TRAIN_DIR), (val, SPLIT_LABELS_VAL_DIR)):
        for index in indices:
            shutil.copy2(paths[index], destination / names[index])
        output = build_unique_stem_index(destination, {".txt"})
        expected = {names[index] for index in indices}
        if {path.name for path in output.values()} != expected:
            raise RuntimeError(f"Split output mismatch: {destination}")

    train_counts, val_counts = counts(classes_per_file, train), counts(classes_per_file, val)
    total_counts = train_counts + val_counts
    split_rows = []
    for split_name, indices in (("train", train), ("val", val)):
        for index in indices:
            split_rows.append({"filename": names[index], "split": split_name, "boxes": len(classes_per_file[index]),
                               "classes": " ".join(map(str, sorted(set(classes_per_file[index]))))})
    pd.DataFrame(split_rows).to_csv(SPLIT_IMAGE_LIST_CSV, index=False)
    pd.DataFrame([{"class_id": c, "train_count": train_counts[c], "val_count": val_counts[c], "total_count": total_counts[c]}
                  for c in range(K)]).to_csv(SPLIT_CLASS_STATS_CSV, index=False)
    train_boxes, val_boxes = sum(train_counts.values()), sum(val_counts.values())
    summary = [
        "MTSD YOLO54 paper2 label-only multilabel 8:2 split summary", "method: MultilabelStratifiedKFold",
        f"seed: {seed}", f"val_ratio_target: {VAL_RATIO}", f"source_labels: {LABELS_FINAL_DIR}",
        f"output_root: {OUTPUT_DIR}", f"usable_label_files: {len(paths)}", f"dropped_empty_label_files: {dropped_empty}",
        f"train_label_files: {len(train)}", f"val_label_files: {len(val)}", f"train_boxes: {train_boxes}",
        f"val_boxes: {val_boxes}", f"val_box_ratio: {val_boxes / (train_boxes + val_boxes):.4f}",
        f"train_class_coverage: {sum(train_counts[c] > 0 for c in range(K))}/{K}",
        f"val_class_coverage: {sum(val_counts[c] > 0 for c in range(K))}/{K}",
        f"split_class_stats: {SPLIT_CLASS_STATS_CSV}", f"split_file_list: {SPLIT_IMAGE_LIST_CSV}",
    ]
    SPLIT_SUMMARY_TXT.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary)); print("Best score:", best_score)


if __name__ == "__main__":
    main()
