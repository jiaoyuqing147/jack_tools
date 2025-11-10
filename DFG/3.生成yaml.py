import os
import re

# ===== 路径配置 =====
classnames_path = r"E:\DataSets\DFG\DFG-tsd-category-info\classnames.txt"  # 类别文件路径
output_yaml_path = r"E:\DataSets\DFG\dfg.yaml"                             # 输出的yaml路径
dataset_root = r"E:\DataSets\DFG\DFG_YOLO"                                 # YOLO格式数据集根目录

# ====================

def safe_read(path):
    """尝试多种编码读取 classnames.txt。"""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.readlines()
        except Exception:
            continue
    raise ValueError("无法读取 classnames.txt，请检查编码")

lines = safe_read(classnames_path)

names = []
for line in lines:
    line = line.strip()
    if not line:
        continue

    # 去掉前缀 "Traffic sign class n - "
    if " - " in line:
        clean = line.split(" - ", 1)[1].strip()
    else:
        clean = line
    names.append(clean)

# 去重并保持顺序
seen = set()
clean_names = []
for n in names:
    if n not in seen:
        seen.add(n)
        clean_names.append(n)

# 写 YAML，names 内无引号
with open(output_yaml_path, "w", encoding="utf-8") as f:
    f.write(f"path: {dataset_root.replace('\\', '/')}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write(f"nc: {len(clean_names)}\n")
    f.write("names: [\n")
    for i, n in enumerate(clean_names):
        sep = "," if i < len(clean_names) - 1 else ""
        f.write(f"  {n}{sep}\n")
    f.write("]\n")

print(f"✅ 已生成 YOLO 数据集配置文件：{output_yaml_path}")
print(f"共 {len(clean_names)} 个类别。")
print("前 5 个类别示例：", clean_names[:5])
