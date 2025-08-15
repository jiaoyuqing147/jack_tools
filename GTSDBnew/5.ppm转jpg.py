import cv2
import os

# ==== 配置路径 ====
source_dir = r'/home/jiaoyuqing/AlgorithmCodes/datasets/GTSDB/yolo/images/test1'
output_dir = r'/home/jiaoyuqing/AlgorithmCodes/datasets/GTSDB/yolo/images/test'

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(source_dir):
    if filename.lower().endswith('.ppm'):
        img_path = os.path.join(source_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"❌ 读取失败: {img_path}")
            continue

        new_filename = os.path.splitext(filename)[0] + '.jpg'
        output_path = os.path.join(output_dir, new_filename)

        # 保存为jpg，参数95是保存质量（可选）
        cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        print(f"✅ 已转换: {filename} -> {new_filename}")

print("🎉 全部 ppm 转换为 jpg 完成！")
