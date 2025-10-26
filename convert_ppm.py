from PIL import Image
import os

# 输入和输出路径
input_folder = r"E:\DataSets\BelgiumTSC\BelgiumTSC\Training\00006"
output_folder = os.path.join(input_folder, "jpg_converted")

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历并转换
for file in os.listdir(input_folder):
    if file.lower().endswith(".ppm"):
        img_path = os.path.join(input_folder, file)
        img = Image.open(img_path)
        img = img.convert("RGB")
        output_path = os.path.join(output_folder, file.replace(".ppm", ".jpg"))
        img.save(output_path, "JPEG")
        print(f"✅ 已转换: {file} → {os.path.basename(output_path)}")

print("全部转换完成！")
