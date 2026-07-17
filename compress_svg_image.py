from pathlib import Path
import re
import base64
from io import BytesIO
from PIL import Image

SVG_PATH = r"C:\Users\JACKJIAO\Desktop\91216_yolo11_train200.svg"
OUT_JPG = r"C:\Users\JACKJIAO\Desktop\91216_yolo11_train200_compressed.jpg"
OUT_SVG = r"C:\Users\JACKJIAO\Desktop\91216_yolo11_train200_compressed.svg"

MAX_IMAGE_WIDTH = 1000
JPEG_QUALITY = 65

svg_text = Path(SVG_PATH).read_text(encoding="utf-8")

pattern = re.compile(
    r'data:image/(?P<fmt>png|jpeg|jpg|webp);base64,(?P<data>[A-Za-z0-9+/=\s]+)',
    re.IGNORECASE
)

match = pattern.search(svg_text)

if not match:
    raise RuntimeError("没有找到 SVG 中的 base64 内嵌图片。")

raw = base64.b64decode(re.sub(r"\s+", "", match.group("data")))

img = Image.open(BytesIO(raw))

if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
    img = img.convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[-1])
    img = bg
else:
    img = img.convert("RGB")

w, h = img.size
print("Original image size:", w, h)

if w > MAX_IMAGE_WIDTH:
    new_w = MAX_IMAGE_WIDTH
    new_h = int(h * new_w / w)
    img = img.resize((new_w, new_h), Image.LANCZOS)

print("New image size:", img.size)

img.save(
    OUT_JPG,
    format="JPEG",
    quality=JPEG_QUALITY,
    optimize=True,
    progressive=True
)

print("Saved JPG:", OUT_JPG)

# 重新生成一个简单 SVG，只包含压缩后的 JPG
buffer = BytesIO()
img.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
jpg_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

new_w, new_h = img.size

new_svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{new_w}"
     height="{new_h}"
     viewBox="0 0 {new_w} {new_h}">
  <image width="{new_w}" height="{new_h}" href="data:image/jpeg;base64,{jpg_b64}"/>
</svg>
'''

Path(OUT_SVG).write_text(new_svg, encoding="utf-8")

print("Saved SVG:", OUT_SVG)
print("JPG size MB:", Path(OUT_JPG).stat().st_size / 1024 / 1024)
print("SVG size MB:", Path(OUT_SVG).stat().st_size / 1024 / 1024)