import os, json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

anno_dir = r"G:\Jack_datasets\Mapillary\Mapillary\mtsd_v2_fully_annotated\annotations"
split_file = r"G:\Jack_datasets\Mapillary\Mapillary\mtsd_v2_fully_annotated\splits\train.txt"

with open(split_file, "r", encoding="utf-8") as f:
    keys = [line.strip() for line in f if line.strip()]

def process_one(key):
    path = os.path.join(anno_dir, key + ".json")
    if not os.path.exists(path):
        return (False, 0)

    data = json.load(open(path, "r", encoding="utf-8"))

    has_occ = False
    occ_cnt = 0

    for obj in data.get("objects", []):
        if obj.get("properties", {}).get("occluded", False):
            has_occ = True
            occ_cnt += 1

    return (has_occ, occ_cnt)

if __name__ == "__main__":
    pool = Pool(cpu_count())

    results = list(tqdm(pool.imap(process_one, keys), total=len(keys)))

    occluded_images = sum(r[0] for r in results)
    occluded_objects = sum(r[1] for r in results)

    print("含遮挡图像数:", occluded_images)
    print("遮挡目标数:", occluded_objects)