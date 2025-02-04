import os
import shutil

# 原始数据文件夹路径
image_folder = r"D:\Jiao\dataset\CatOnlyCOCOVOC\Dark\images"
annotation_folder = r"D:\Jiao\dataset\CatOnlyCOCOVOC\annotations\xmlall"
destination_folder = r"D:\Jiao\dataset\CatOnlyCOCOVOC\Dark\annotations"

# 创建目标文件夹（如果不存在）
os.makedirs(destination_folder, exist_ok=True)

# 获取train2017Dark中的所有图片编号（去掉.jpg后缀）
image_files = [f.split('.')[0] for f in os.listdir(image_folder) if f.endswith('.jpg')]

# 遍历标注文件夹，将匹配的xml文件复制到新文件夹
for xml_file in os.listdir(annotation_folder):
    if xml_file.endswith('.xml'):
        file_id = xml_file.split('.')[0]  # 提取XML文件编号
        if file_id in image_files:  # 检查是否在目标图像列表中
            src_path = os.path.join(annotation_folder, xml_file)
            dst_path = os.path.join(destination_folder, xml_file)
            shutil.copy2(src_path, dst_path)  # 复制文件

print("XML 文件复制完成！")

