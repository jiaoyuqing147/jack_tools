import json
from collections import Counter

json_path = r"E:\Datasets\tt100k_2021\annotations_all.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

types = data["types"]
type_to_id = {name: idx for idx, name in enumerate(types)}

train_counter = Counter()
test_counter = Counter()
train_images = 0
test_images = 0

for img_id, img_data in data["imgs"].items():
    rel_path = img_data["path"]
    subset = rel_path.split("/")[0]

    if subset == "train":
        train_images += 1
        counter = train_counter
    elif subset == "test":
        test_images += 1
        counter = test_counter
    else:
        continue

    for obj in img_data["objects"]:
        category = obj["category"]
        if category in type_to_id:
            counter[type_to_id[category]] += 1

train_ids = set(train_counter.keys())
test_ids = set(test_counter.keys())
common_ids = train_ids & test_ids
train_only_ids = train_ids - test_ids
test_only_ids = test_ids - train_ids

print("===== 数据统计 =====")
print(f"train 图片数: {train_images}")
print(f"test 图片数 : {test_images}")
print(f"train 目标框数: {sum(train_counter.values())}")
print(f"test 目标框数 : {sum(test_counter.values())}")

print("\n===== 类别覆盖 =====")
print(f"train 出现类别数: {len(train_ids)}")
print(f"test 出现类别数 : {len(test_ids)}")
print(f"共同类别数      : {len(common_ids)}")
print(f"只在 train 中出现: {len(train_only_ids)}")
print(f"只在 test 中出现 : {len(test_only_ids)}")

print("\n===== 共同类别，按总目标数排序 =====")
for cid in sorted(common_ids, key=lambda x: train_counter[x] + test_counter[x], reverse=True):
    print(
        f"{cid:3d} {types[cid]:10s} "
        f"train={train_counter[cid]:5d} "
        f"test={test_counter[cid]:5d} "
        f"total={train_counter[cid] + test_counter[cid]:5d}"
    )

print("\n===== 只在 train 中出现的类别 =====")
for cid in sorted(train_only_ids):
    print(f"{cid:3d} {types[cid]:10s} train={train_counter[cid]}")

print("\n===== 只在 test 中出现的类别 =====")
for cid in sorted(test_only_ids):
    print(f"{cid:3d} {types[cid]:10s} test={test_counter[cid]}")