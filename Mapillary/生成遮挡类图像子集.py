import os
import json
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ======================
# 路径配置
# ======================
root = r"G:\Jack_datasets\Mapillary\Mapillary\mtsd_v2_fully_annotated"
anno_dir = os.path.join(root, "annotations")
split_dir = os.path.join(root, "splits")
out_dir = os.path.join(root, "splits_occluded")
os.makedirs(out_dir, exist_ok=True)

# 每个 split 抽多少张
sample_nums = {
    "train": 2000,
    "val": 500,
    "test": 500,
}

random.seed(42)


def read_keys(split_name):
    txt_path = os.path.join(split_dir, f"{split_name}.txt")
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_txt(keys, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for k in keys:
            f.write(k + "\n")


def check_occluded(key):
    json_path = os.path.join(anno_dir, key + ".json")

    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for obj in data.get("objects", []):
            props = obj.get("properties", {})
            if props.get("occluded") is True:
                return key

        return None

    except Exception as e:
        print(f"读取失败: {json_path}, {e}")
        return None


def process_split(split_name):
    keys = read_keys(split_name)
    print(f"\n========== {split_name} ==========")
    print(f"原始图像数: {len(keys)}")

    # 多进程筛选
    workers = max(1, cpu_count() - 1)
    with Pool(workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(check_occluded, keys, chunksize=200),
                total=len(keys),
                desc=f"筛选 {split_name}"
            )
        )

    occluded_keys = [r for r in results if r is not None]

    print(f"含遮挡图像数: {len(occluded_keys)}")

    # 保存全部遮挡图像
    all_path = os.path.join(out_dir, f"{split_name}_occluded_all.txt")
    save_txt(occluded_keys, all_path)

    # 抽样保存
    n = sample_nums.get(split_name, 500)
    sample_keys = random.sample(occluded_keys, min(n, len(occluded_keys)))

    sample_path = os.path.join(out_dir, f"{split_name}_occluded_{len(sample_keys)}.txt")
    save_txt(sample_keys, sample_path)

    print("已保存:")
    print(all_path)
    print(sample_path)


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        process_split(split)

    print("\n全部完成！输出目录：")
    print(out_dir)