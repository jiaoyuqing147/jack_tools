import cv2
from pathlib import Path

# =========================================================
# 输入输出文件夹
# =========================================================
input_dir = Path(r"C:\Users\JACKJIAO\Desktop\00239Heatmap")
output_dir = Path(r"C:\Users\JACKJIAO\Desktop\00239Heatmap_resize1280")

# 如果输出目录不存在，就创建
output_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# 支持的图像格式
# =========================================================
img_suffixes = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# =========================================================
# 遍历所有图像
# =========================================================
count = 0

for img_path in input_dir.iterdir():
    if img_path.suffix.lower() not in img_suffixes:
        continue

    img = cv2.imread(str(img_path))

    if img is None:
        print(f"❌ 读取失败: {img_path}")
        continue

    # =====================================================
    # 强制拉伸到 1280×1280
    # =====================================================
    img_resize = cv2.resize(img, (1280, 1280), interpolation=cv2.INTER_LANCZOS4)

    # 输出路径（文件名不变）
    save_path = output_dir / img_path.name

    cv2.imwrite(str(save_path), img_resize)

    count += 1
    print(f"✅ 已处理: {img_path.name}")

print(f"\n🎉 完成，共处理 {count} 张图像")