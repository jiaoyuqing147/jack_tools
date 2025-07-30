import os
import shutil

# === 设置路径 ===
image_dir = r"/home/jiaoyuqing/bigspace/workspaceJack/datasets/TT100K/tt100k_2021/yolo143/images/test"
label_dir = r"/home/jiaoyuqing/bigspace/workspaceJack/datasets/TT100K/tt100k_2021/yolo143/labels/test"
unused_dir = r"/home/jiaoyuqing/bigspace/workspaceJack/datasets/TT100K/tt100k_2021/yolo143/images/test_unused"

# 创建未配对图像存放目录
os.makedirs(unused_dir, exist_ok=True)

# 收集标签 base 名称（不含扩展名，转为小写）
label_names = set(os.path.splitext(name)[0].lower() for name in os.listdir(label_dir) if name.endswith(".txt"))

# 统计移动记录
moved_files = []

# 遍历图像文件
for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    base_name = os.path.splitext(img_name)[0].lower()
    if base_name not in label_names:
        # 没有对应标签文件，移动到 unused 文件夹
        src = os.path.join(image_dir, img_name)
        dst = os.path.join(unused_dir, img_name)
        shutil.move(src, dst)
        moved_files.append(img_name)
        print(f"[已移动] 未配对图像：{img_name}")

# 输出统计
print(f"\n✅ 共移动未匹配图像：{len(moved_files)} 张 到：{unused_dir}")

# 可选：打印移动的文件列表
if moved_files:
    print("\n📋 被移动的图像列表：")
    for f in moved_files:
        print(" -", f)
