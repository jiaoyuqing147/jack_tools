import base64
from pathlib import Path
from PIL import Image

# ==============================
# 输入文件夹
# ==============================
img_dir = Path(r"C:\Users\JACKJIAO\Desktop\00239Heatmap_resize1280")

if not img_dir.exists():
    raise FileNotFoundError(f"文件夹不存在: {img_dir}")

# ==============================
# PNG 转 SVG
# ==============================
def png_to_svg(png_path: Path):
    with Image.open(png_path) as im:
        width, height = im.size

    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    data_uri = f"data:image/png;base64,{b64}"

    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <image href="{data_uri}" x="0" y="0" width="{width}" height="{height}"/>
</svg>
'''

    svg_path = png_path.with_suffix(".svg")
    svg_path.write_text(svg_content, encoding="utf-8")

    print(f"✅ 已转换: {png_path.name} -> {svg_path.name}")

# ==============================
# 批量处理
# ==============================
png_files = list(img_dir.glob("*.png"))

if not png_files:
    print("⚠️ 文件夹中没有找到 PNG 图片")
else:
    for png_file in png_files:
        png_to_svg(png_file)

    print(f"\n🎉 全部完成，共转换 {len(png_files)} 张 PNG 为 SVG")