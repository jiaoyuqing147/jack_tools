import os
import glob
import shutil
from typing import Union, Optional

# ===== 路径配置 =====
labels_root = r"/home/jiaoyuqing/bigspace/workspaceJack/datasets/MTSD/MTSD/yolo54/labels"
images_src_root = r"/home/jiaoyuqing/bigspace/workspaceJack/datasets/MTSD/MTSD/Detection"
images_dst_root = r"/home/jiaoyuqing/bigspace/workspaceJack/datasets/MTSD/MTSD/yolo54/images"

# 可能的图片扩展名（小写形式，用于匹配）
valid_exts = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".gif"]

splits = ["train", "val", "test"]

# ===== 创建目标目录 =====
for split in splits:
    os.makedirs(os.path.join(images_dst_root, split), exist_ok=True)


def find_matching_image(txt_stem: str) -> Union[str, None]:
    """
    查找与标签文件匹配的图像，忽略大小写
    将标签文件名转为小写后与图像文件名（小写）匹配
    """
    target_stem = txt_stem.lower()  # 标签文件名转为小写作为基准

    # 遍历源目录所有文件，检查是否匹配
    for filename in os.listdir(images_src_root):
        file_stem, file_ext = os.path.splitext(filename)
        # 文件名和扩展名均转为小写后匹配
        if file_stem.lower() == target_stem and file_ext.lower() in valid_exts:
            return os.path.join(images_src_root, filename)

    return None


def process_split(split: str):
    label_dir = os.path.join(labels_root, split)
    dest_dir = os.path.join(images_dst_root, split)

    # 获取所有标签文件
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
    if not txt_files:
        print(f"⚠️ 在 {label_dir} 未找到任何TXT文件")
        return

    success = 0
    missing = 0
    missing_files = []

    for txt_path in txt_files:
        # 提取标签文件名（不含路径和扩展名）
        txt_filename = os.path.basename(txt_path)
        txt_stem = os.path.splitext(txt_filename)[0]

        # 查找匹配的图片
        img_path = find_matching_image(txt_stem)

        if img_path:
            # 构建目标文件名：文件名和后缀名均转为小写
            src_filename = os.path.basename(img_path)
            src_stem, src_ext = os.path.splitext(src_filename)
            dest_filename = f"{src_stem.lower()}{src_ext.lower()}"  # 全小写处理
            dest_path = os.path.join(dest_dir, dest_filename)

            # 复制文件到目标目录
            shutil.copy2(img_path, dest_path)
            success += 1
        else:
            missing += 1
            missing_files.append(txt_stem)

    # 输出处理结果
    print(f"✅ [{split}] 成功匹配并复制: {success} 个文件")
    print(f"❌ [{split}] 未找到对应图片: {missing} 个文件")

    # 打印部分未找到的文件示例
    if missing_files:
        preview = ", ".join(missing_files[:10])
        more = "..." if len(missing_files) > 10 else ""
        print(f"  未找到的示例: {preview}{more}")
    print("-" * 50)


# ===== 执行主程序 =====
if __name__ == "__main__":
    print("开始处理图片匹配与复制...")
    print(f"图片源目录: {images_src_root}")
    print(f"目标目录: {images_dst_root}\n")

    for split in splits:
        process_split(split)

    print("处理完成！所有复制的图像文件均已转为小写文件名和小写后缀名")