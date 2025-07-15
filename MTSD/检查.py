import pandas as pd

gt_df = pd.read_csv('GT_Detection.txt', delimiter=';', encoding='latin1')
gt_files = gt_df['File Name'].str.strip().str.strip("'").str.lower().unique()

image_size_df = pd.read_csv('image_sizes.csv')
image_files = image_size_df['filename'].str.strip().str.strip("'").str.lower().unique()

# 图片总数
print(f"图片总数：{len(image_files)}")
# 标注文件关联的图片数
print(f"GT标注涉及的图片数：{len(gt_files)}")

# 统计哪些图片无标注
images_without_labels = set(image_files) - set(gt_files)
print(f"\n没有标注的图片数量：{len(images_without_labels)}")
for img in sorted(images_without_labels):
    print(img)
