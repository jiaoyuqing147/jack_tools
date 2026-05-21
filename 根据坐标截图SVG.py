import re
from pathlib import Path

# ========= 输入文件 =========
# svg_path = Path(r"C:\Users\JACKJIAO\Desktop\23858\23858_yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation.svg")

# svg_path = Path(r"C:\Users\JACKJIAO\Desktop\23858heatmap\23858_yolo11_train200.svg")
svg_path = Path(r"C:\Users\JACKJIAO\Desktop\对比实验\12467_yolo11_train200.svg")


x1, y1 = 737, 551
x2, y2 = 1665, 1480
# ========= 裁剪坐标 =========
# x1, y1 = 1297, 922
# x2, y2 = 1343, 968
#
# x1, y1 = 317, 1006
# x2, y2 = 355, 1044


crop_w = x2 - x1
crop_h = y2 - y1

if crop_w <= 0 or crop_h <= 0:
    raise ValueError("裁剪区域无效：右下角必须大于左上角")

# ========= 输出路径（保存到同文件夹）=========
output_path = svg_path.parent / f"{svg_path.stem}_crop.svg"

# ========= 读取SVG =========
text = svg_path.read_text(encoding="utf-8")

# ========= 检查原始画布范围 =========
match = re.search(r'viewBox="([^"]+)"', text)
if match:
    vals = list(map(float, match.group(1).split()))
    if len(vals) != 4:
        raise ValueError("viewBox 格式异常")
    vb_x, vb_y, vb_w, vb_h = vals
else:
    w_match = re.search(r'width="([\d.]+)"', text)
    h_match = re.search(r'height="([\d.]+)"', text)
    if not w_match or not h_match:
        raise ValueError("未找到 viewBox 或 width/height")
    vb_x, vb_y = 0.0, 0.0
    vb_w = float(w_match.group(1))
    vb_h = float(h_match.group(1))

# ========= 防越界 =========
crop_x = max(vb_x, x1)
crop_y = max(vb_y, y1)
crop_x2 = min(vb_x + vb_w, x2)
crop_y2 = min(vb_y + vb_h, y2)

crop_w = crop_x2 - crop_x
crop_h = crop_y2 - crop_y

if crop_w <= 0 or crop_h <= 0:
    raise ValueError(
        f"裁剪区域超出图像范围。\n"
        f"原始范围: ({vb_x}, {vb_y}, {vb_x + vb_w}, {vb_y + vb_h})\n"
        f"请求范围: ({x1}, {y1}, {x2}, {y2})"
    )

# ========= 写入新的 viewBox =========
new_viewbox = f"{crop_x:.4f} {crop_y:.4f} {crop_w:.4f} {crop_h:.4f}"

if 'viewBox="' in text:
    text = re.sub(r'viewBox="[^"]+"', f'viewBox="{new_viewbox}"', text, count=1)
else:
    text = text.replace("<svg ", f'<svg viewBox="{new_viewbox}" ', 1)

# 同时改 width / height，避免显示时还是原尺寸
if re.search(r'width="[^"]+"', text):
    text = re.sub(r'width="[^"]+"', f'width="{crop_w:.4f}"', text, count=1)
else:
    text = text.replace("<svg ", f'<svg width="{crop_w:.4f}" ', 1)

if re.search(r'height="[^"]+"', text):
    text = re.sub(r'height="[^"]+"', f'height="{crop_h:.4f}"', text, count=1)
else:
    text = text.replace("<svg ", f'<svg height="{crop_h:.4f}" ', 1)

# ========= 保存 =========
output_path.write_text(text, encoding="utf-8")

print(f"✅ 已裁剪完成")
print(f"输入文件: {svg_path}")
print(f"输出文件: {output_path}")
print(f"裁剪区域: ({crop_x}, {crop_y}) -> ({crop_x2}, {crop_y2})")
print(f"裁剪尺寸: {crop_w} x {crop_h}")