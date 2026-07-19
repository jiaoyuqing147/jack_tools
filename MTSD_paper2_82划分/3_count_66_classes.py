import pandas as pd

from mtsd_paths_config import CLASS_COUNTS_CSV, GT_RECOGNITION, make_output_dirs


def normalize_name(value: object) -> str:
    return str(value).strip().strip("'").strip('"').lower()


def main() -> None:
    make_output_dirs()

    recognition_df = pd.read_csv(GT_RECOGNITION, sep=";", engine="python")
    recognition_df["File Name"] = recognition_df["File Name"].map(normalize_name)

    class_counts = recognition_df["Class ID"].astype(int).value_counts().sort_index()
    class_counts.to_csv(CLASS_COUNTS_CSV, header=["count"])

    for class_id, count in class_counts.items():
        print(f"Class ID {class_id}: {count} 次")

    print(f"\nSaved class counts: {CLASS_COUNTS_CSV}")


if __name__ == "__main__":
    main()
