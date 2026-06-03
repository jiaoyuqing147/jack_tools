from pathlib import Path
import shutil

# ==============================
# 路径
# ==============================

txt_path = Path(
    r"E:\DataSets\tt100k_2021_paper2\tt100k_60_weather\normal.txt"
)

src_dir = Path(
    r"E:\DataSets\tt100k_2021_paper2\tt100k_60\images\train"
)

dst_dir = Path(
    r"E:\DataSets\tt100k_2021_paper2\tt100k_60_weather\normal"
)

# 创建目标文件夹
dst_dir.mkdir(parents=True, exist_ok=True)

# ==============================
# 读取编号
# ==============================

with open(txt_path, "r", encoding="utf-8") as f:
    ids = [line.strip() for line in f if line.strip()]

# ==============================
# 复制图片
# ==============================

count = 0
missing = []

for img_id in ids:

    # 拼接 jpg 文件名
    src_path = src_dir / f"{img_id}.jpg"

    # 如果是 png 可以改成 .png
    # src_path = src_dir / f"{img_id}.png"

    if src_path.exists():

        dst_path = dst_dir / src_path.name

        shutil.copy2(src_path, dst_path)

        count += 1

    else:
        missing.append(img_id)

# ==============================
# 输出结果
# ==============================

print(f"成功复制 {count} 张图像")

if missing:
    print("\n以下文件不存在：")

    for x in missing:
        print(x)