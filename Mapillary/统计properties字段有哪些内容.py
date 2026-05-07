import os
import json
from collections import defaultdict, Counter
from tqdm import tqdm

# ======================
# 路径配置（改成你的）
# ======================
anno_dir = r"G:\Jack_datasets\Mapillary\Mapillary\mtsd_v2_fully_annotated\annotations"
split_file = r"G:\Jack_datasets\Mapillary\Mapillary\mtsd_v2_fully_annotated\splits\train.txt"

# ======================
# 读取 split（只统计train）
# ======================
with open(split_file, "r") as f:
    image_keys = [line.strip() for line in f]

print(f"共 {len(image_keys)} 张图（train）")

# ======================
# 统计
# ======================
prop_counter = Counter()  # 每个属性出现次数
prop_value_counter = defaultdict(Counter)  # 每个属性的取值分布

total_objects = 0
has_properties = 0

for key in tqdm(image_keys):
    json_path = os.path.join(anno_dir, key + ".json")

    if not os.path.exists(json_path):
        continue

    data = json.load(open(json_path, "r"))

    for obj in data["objects"]:
        total_objects += 1

        props = obj.get("properties", {})
        if len(props) > 0:
            has_properties += 1

        for k, v in props.items():
            prop_counter[k] += 1
            prop_value_counter[k][str(v)] += 1

# ======================
# 输出结果
# ======================
print("\n====================")
print("总目标数:", total_objects)
print("带properties的目标数:", has_properties)
print("====================\n")

print("=== 属性字段统计 ===")
for k, v in prop_counter.items():
    print(f"{k}: {v}")

print("\n=== 属性取值分布 ===")
for k, counter in prop_value_counter.items():
    print(f"\n{k}:")
    for val, cnt in counter.most_common(10):
        print(f"  {val}: {cnt}")
