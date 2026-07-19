from pathlib import Path

import pandas as pd

from mtsd_case_utils import (
    IMAGE_SUFFIXES, assert_exact_output, build_unique_stem_index, clear_owned_files,
    validate_yolo_file,
)
from mtsd_paths_config import (
    CLASS_COUNTS_CSV, DROP_MAX_COUNT, IMAGE_DIR, KEPT_IDS_TXT, LABELS_FINAL_DIR,
    LABELS_PRUNED_FILTERED_DIR, LABELS_YOLO66_DIR, MAPPING_CSV, ORIGINAL_CLASS_COUNT,
    REMOVED_CLASS_IDS_1BASED_TXT, REMOVED_IDS_TXT, SUMMARY_TXT, make_output_dirs,
)


def canonical_sources() -> list[tuple[Path, str]]:
    images = build_unique_stem_index(IMAGE_DIR, IMAGE_SUFFIXES)
    labels = build_unique_stem_index(LABELS_YOLO66_DIR, {".txt"})
    result = []
    for key, label in labels.items():
        image = images.get(key)
        if image is None:
            raise FileNotFoundError(f"YOLO66 label has no source image: {label}")
        result.append((label, f"{image.stem}.txt"))
    return sorted(result, key=lambda item: (item[1].casefold(), item[1]))


def main() -> None:
    make_output_dirs()
    if not CLASS_COUNTS_CSV.is_file():
        raise FileNotFoundError(CLASS_COUNTS_CSV)
    sources = canonical_sources()
    for source, _ in sources:
        validate_yolo_file(source, ORIGINAL_CLASS_COUNT)

    counts = pd.read_csv(CLASS_COUNTS_CSV, index_col=0)
    if "count" not in counts.columns:
        counts.columns = ["count"]
    counts.index = counts.index.astype(int)
    low_1based = counts[counts["count"] <= DROP_MAX_COUNT].index.tolist()
    low_0based = sorted(class_id - 1 for class_id in low_1based)

    clear_owned_files(LABELS_PRUNED_FILTERED_DIR, {".txt"})
    # LABELS_FINAL_DIR also contains train/val subdirs. This step owns only root .txt files.
    LABELS_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for old_label in LABELS_FINAL_DIR.glob("*.txt"):
        old_label.unlink()
    if list(LABELS_FINAL_DIR.glob("*.txt")):
        raise RuntimeError(f"Failed to clear old root labels in {LABELS_FINAL_DIR}")
    total_removed = 0
    files_changed = 0
    expected = [name for _, name in sources]
    for source, output_name in sources:
        kept = []
        removed = 0
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if int(line.split()[0]) in low_0based:
                removed += 1
            else:
                kept.append(line.strip())
        output = LABELS_PRUNED_FILTERED_DIR / output_name
        output.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        validate_yolo_file(output, ORIGINAL_CLASS_COUNT)
        total_removed += removed
        files_changed += int(removed > 0)
    assert_exact_output(LABELS_PRUNED_FILTERED_DIR, expected)

    kept_old_ids = sorted({
        int(line.split()[0])
        for path in LABELS_PRUNED_FILTERED_DIR.glob("*.txt")
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    })
    removed_old_ids = sorted(set(range(ORIGINAL_CLASS_COUNT)) - set(kept_old_ids))
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(kept_old_ids)}
    pd.DataFrame([
        {"old_id": old_id, "new_id": new_id, "old_class_id_1based": old_id + 1}
        for old_id, new_id in old_to_new.items()
    ]).to_csv(MAPPING_CSV, index=False)
    KEPT_IDS_TXT.write_text("\n".join(map(str, kept_old_ids)), encoding="utf-8")
    REMOVED_IDS_TXT.write_text("\n".join(map(str, removed_old_ids)), encoding="utf-8")
    REMOVED_CLASS_IDS_1BASED_TXT.write_text("\n".join(str(v + 1) for v in removed_old_ids), encoding="utf-8")

    final_boxes = 0
    empty_files = 0
    for source, output_name in [(LABELS_PRUNED_FILTERED_DIR / name, name) for name in expected]:
        remapped = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                parts = line.split()
                parts[0] = str(old_to_new[int(parts[0])])
                remapped.append(" ".join(parts))
        output = LABELS_FINAL_DIR / output_name
        output.write_text("\n".join(remapped) + ("\n" if remapped else ""), encoding="utf-8")
        validate_yolo_file(output, len(kept_old_ids))
        final_boxes += len(remapped)
        empty_files += int(not remapped)
    assert_exact_output(LABELS_FINAL_DIR, expected)
    summary = [
        "MTSD prune and remap summary", f"drop_max_count: {DROP_MAX_COUNT}",
        f"removed_class_count: {len(removed_old_ids)}", f"kept_class_count: {len(kept_old_ids)}",
        f"files_with_removed_boxes: {files_changed}", f"removed_boxes: {total_removed}",
        f"final_label_files: {len(expected)}", f"final_boxes: {final_boxes}", f"empty_label_files: {empty_files}",
    ]
    SUMMARY_TXT.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
