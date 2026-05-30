from collections import Counter
import os

K = 43

root = r"E:\DataSets\GTSDB\yolo43\labels"

for split in ["train", "val", "test"]:

    cnt = Counter()

    split_dir = os.path.join(root, split)

    for txt in os.listdir(split_dir):

        txt_path = os.path.join(split_dir, txt)

        with open(txt_path, "r") as f:

            for line in f:

                cls = int(line.split()[0])

                cnt[cls] += 1

    print(f"\n===== {split} =====")

    print("num_classes =", len(cnt))
    print("num_boxes =", sum(cnt.values()))

    for k in range(K):
        print(k, cnt[k])