import json, os

# 路径配置
json_path = r"E:\DataSets\DFG\DFG-tsd-annot-json\train.json"
img_dir = r"E:\DataSets\DFG\JPEGImages"

data = json.load(open(json_path, "r"))
image_files = {img["id"]: img["file_name"] for img in data["images"]}

print(f"共有 {len(image_files)} 张图片")

# 查看第一张图片对应的标注框数
first_img_id = list(image_files.keys())[0]
annots = [a for a in data["annotations"] if a["image_id"] == first_img_id]
print(f"图片 {image_files[first_img_id]} 有 {len(annots)} 个标注")
