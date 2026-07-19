import pandas as pd

from mtsd_paths_config import (
    CLASS_COUNTS_CSV,
    DROP_MAX_COUNT,
    KEPT_IDS_TXT,
    LABELS_FINAL_DIR,
    LABELS_PRUNED_FILTERED_DIR,
    LABELS_YOLO66_DIR,
    MAPPING_CSV,
    ORIGINAL_CLASS_COUNT,
    REMOVED_CLASS_IDS_1BASED_TXT,
    REMOVED_IDS_TXT,
    SUMMARY_TXT,
    make_output_dirs,
)


def main() -> None:
    make_output_dirs()

    counts = pd.read_csv(CLASS_COUNTS_CSV, index_col=0)
    if "count" not in counts.columns:
        counts.columns = ["count"]
    counts.index = counts.index.astype(int)

    low_freq_1based = counts[counts["count"] <= DROP_MAX_COUNT].index.tolist()
    low_freq_0based = sorted(class_id - 1 for class_id in low_freq_1based)
    print(f"将删除的类别，原始 Class ID 1-based: {low_freq_1based}")
    print(f"将删除的类别，YOLO old_id 0-based: {low_freq_0based}")

    total_removed = 0
    files_changed = 0

    for src in sorted(LABELS_YOLO66_DIR.glob("*.txt")):
        lines = [line.strip() for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]

        filtered_lines = []
        removed = 0
        for line in lines:
            parts = line.split()
            old_id = int(parts[0])
            if old_id in low_freq_0based:
                removed += 1
            else:
                filtered_lines.append(line)

        dst = LABELS_PRUNED_FILTERED_DIR / src.name
        dst.write_text("\n".join(filtered_lines) + ("\n" if filtered_lines else ""), encoding="utf-8")

        if removed:
            files_changed += 1
            total_removed += removed

    kept_old_ids = set()
    for src in sorted(LABELS_PRUNED_FILTERED_DIR.glob("*.txt")):
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                kept_old_ids.add(int(line.split()[0]))

    kept_old_ids = sorted(kept_old_ids)
    removed_old_ids = sorted(set(range(ORIGINAL_CLASS_COUNT)) - set(kept_old_ids))
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(kept_old_ids)}

    pd.DataFrame(
        [{"old_id": old_id, "new_id": new_id, "old_class_id_1based": old_id + 1}
         for old_id, new_id in old_to_new.items()]
    ).to_csv(MAPPING_CSV, index=False)

    KEPT_IDS_TXT.write_text("\n".join(map(str, kept_old_ids)), encoding="utf-8")
    REMOVED_IDS_TXT.write_text("\n".join(map(str, removed_old_ids)), encoding="utf-8")
    REMOVED_CLASS_IDS_1BASED_TXT.write_text(
        "\n".join(map(str, [old_id + 1 for old_id in removed_old_ids])), encoding="utf-8"
    )

    final_files = 0
    final_boxes = 0
    empty_files = 0

    for src in sorted(LABELS_PRUNED_FILTERED_DIR.glob("*.txt")):
        remapped_lines = []

        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            old_id = int(parts[0])
            parts[0] = str(old_to_new[old_id])
            remapped_lines.append(" ".join(parts))

        dst = LABELS_FINAL_DIR / src.name
        dst.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8")

        final_files += 1
        final_boxes += len(remapped_lines)
        if not remapped_lines:
            empty_files += 1

    summary = [
        "MTSD prune and remap summary",
        f"drop_max_count: {DROP_MAX_COUNT}",
        f"removed_class_count: {len(removed_old_ids)}",
        f"kept_class_count: {len(kept_old_ids)}",
        f"files_with_removed_boxes: {files_changed}",
        f"removed_boxes: {total_removed}",
        f"final_label_files: {final_files}",
        f"final_boxes: {final_boxes}",
        f"empty_label_files: {empty_files}",
    ]
    SUMMARY_TXT.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Filtered labels dir: {LABELS_PRUNED_FILTERED_DIR}")
    print(f"Final labels dir: {LABELS_FINAL_DIR}")
    print(f"Mapping csv: {MAPPING_CSV}")
    print(f"Kept classes: {len(kept_old_ids)}")
    print(f"Removed classes: {len(removed_old_ids)}")
    print(f"Removed boxes: {total_removed}")


if __name__ == "__main__":
    main()
