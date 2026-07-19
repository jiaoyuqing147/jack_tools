# MTSD YOLO54 Scripts

These scripts keep the original five-step workflow, but all intermediate files are written into one clean output tree.

## Run Order

```bash
python 1_get_image_sizes.py
python 2_merge_detection_recognition_to_yolo66.py
python 3_count_66_classes.py
python 4_prune_le3_and_remap_labels.py
python 5_generate_classes54.py
```

## Output Tree

```text
E:\DataSets\MTSD\yolo54\
  labels\                         final YOLO labels, remapped to 54 classes
  classes.txt                     final class names
  names.yaml                      final YAML names mapping
  intermediate\
    labels_yolo66\                temporary original 66-class YOLO labels
    labels_pruned_filtered\       temporary labels after deleting classes <= 3
  reports\
    image_sizes.csv
    class_id_counts.csv
    class_id_mapping.csv
    kept_class_ids_old_0based.txt
    removed_class_ids_old_0based.txt
    removed_class_ids_original_1based.txt
    missing_recognition.txt
    missing_image_size.txt
    detection_recognition_mismatches.csv
    summary.txt
```

Only `yolo54/labels`, `yolo54/classes.txt`, and `yolo54/names.yaml` are needed for training.
