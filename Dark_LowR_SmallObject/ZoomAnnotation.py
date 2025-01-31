'''
这个脚本会： ✅ 读取原始标注文件
✅ 更新 width 和 height（乘以 2）
✅ 保存到新的 labels_enhanced 目录
✅ 保持原文件名不变
'''

import os

# 原始标注文件夹（输入文件夹）
label_dir = r"D:\Jiao\dataset\Jack_generate_cat\DarkResizeYolostyle\labels\val2017"

# 新的标注文件夹（输出文件夹）
output_label_dir = r"D:\Jiao\dataset\Jack_generate_cat\DarkResizeYolostyle\labels\val2017_enhanced"

# 确保新文件夹存在
os.makedirs(output_label_dir, exist_ok=True)

# 遍历所有标注文件
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):  # 只处理文本文件
        input_file_path = os.path.join(label_dir, filename)  # 原始标注文件
        output_file_path = os.path.join(output_label_dir, filename)  # 新标注文件

        # 读取原始标注数据
        with open(input_file_path, "r") as f:
            lines = f.readlines()

        # 处理每一行的标注信息
        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = parts[0]
                x_center = float(parts[1])  # 归一化的中心 x 坐标（不变）
                y_center = float(parts[2])  # 归一化的中心 y 坐标（不变）
                width = float(parts[3]) * 2  # 放大 2 倍
                height = float(parts[4]) * 2  # 放大 2 倍

                # 重新组合数据
                updated_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                updated_lines.append(updated_line)

        # 写入新的标注文件（不覆盖原文件）
        with open(output_file_path, "w") as f:
            f.writelines(updated_lines)

print("所有标注已更新完成，并保存到新文件夹！🚀")
