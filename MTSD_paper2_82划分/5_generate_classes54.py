import pandas as pd

from mtsd_paths_config import CLASSES54_TXT, CLASSES66_TXT, MAPPING_CSV, NAMES54_YAML, make_output_dirs


def find_classes66_txt():
    candidates = [
        CLASSES66_TXT,
        MAPPING_CSV.parents[1] / "classes66.txt",
        MAPPING_CSV.parents[1].parent / "classes66.txt",
        __import__("pathlib").Path(__file__).resolve().parent / "classes66.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "找不到 classes66.txt。请把 66 类名称文件放到以下任意位置：\n"
        f"1) {CLASSES66_TXT}\n"
        f"2) {__import__('pathlib').Path(__file__).resolve().parent / 'classes66.txt'}\n"
        "要求：每行一个类别名，第1行对应原始 Class ID 1，第66行对应原始 Class ID 66。"
    )


def main() -> None:
    make_output_dirs()

    mapping_df = pd.read_csv(MAPPING_CSV)
    old2new = dict(zip(mapping_df["old_id"].astype(int), mapping_df["new_id"].astype(int)))

    classes66_path = find_classes66_txt()
    names66 = [line.strip() for line in classes66_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names66) < max(old2new) + 1:
        raise ValueError(f"{classes66_path} has {len(names66)} names, but mapping needs index {max(old2new)}")

    pairs = [(new_id, names66[old_id]) for old_id, new_id in old2new.items()]
    pairs.sort(key=lambda item: item[0])
    names54 = [name for _, name in pairs]

    CLASSES54_TXT.write_text("\n".join(names54) + "\n", encoding="utf-8")
    NAMES54_YAML.write_text(
        "names:\n" + "".join(f'  {i}: "{name}"\n' for i, name in enumerate(names54)),
        encoding="utf-8",
    )

    print(f"Saved classes.txt: {CLASSES54_TXT}")
    print(f"Saved names.yaml: {NAMES54_YAML}")
    print(f"Source classes66.txt: {classes66_path}")
    print(f"Class count: {len(names54)}")


if __name__ == "__main__":
    main()
