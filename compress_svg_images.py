

from pathlib import Path
import re
import base64
from io import BytesIO
from PIL import Image


# =====================================================
# 1. 文件夹路径
# =====================================================
SVG_DIR = r"C:\Users\JACKJIAO\Desktop\12"
OUTPUT_DIR = r"C:\Users\JACKJIAO\Desktop\12_compressed"


# =====================================================
# 2. 压缩参数
# =====================================================
# 每个 SVG 内嵌底图的最大宽度
# 建议：
# 1200：文件更小
# 1400：比较均衡
# 1600：更清楚但文件更大
MAX_IMAGE_WIDTH = 800

# JPEG 压缩质量
# 建议：
# 80：文件更小
# 85：比较均衡
# 88~90：更清楚但文件更大
JPEG_QUALITY = 65


# =====================================================
# 3. 图片压缩函数
# =====================================================
def compress_image_bytes(image_bytes, max_width=1400, jpeg_quality=85):
    """
    压缩 SVG 中 base64 内嵌的图片。

    处理逻辑：
    1. 打开图片；
    2. 如果图片宽度大于 max_width，则等比例缩小；
    3. 转为 RGB JPEG；
    4. 返回新的 data:image/jpeg;base64,... 字符串。
    """
    img = Image.open(BytesIO(image_bytes))

    original_size = img.size

    # 处理透明背景，避免转 JPEG 后变黑
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    else:
        img = img.convert("RGB")

    w, h = img.size

    # 按最大宽度等比例缩小
    if w > max_width:
        new_w = max_width
        new_h = int(h * new_w / w)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    new_size = img.size

    out = BytesIO()
    img.save(
        out,
        format="JPEG",
        quality=jpeg_quality,
        optimize=True,
        progressive=True
    )

    compressed_bytes = out.getvalue()
    encoded = base64.b64encode(compressed_bytes).decode("ascii")

    new_data_uri = f"data:image/jpeg;base64,{encoded}"

    return new_data_uri, original_size, new_size, len(compressed_bytes)


# =====================================================
# 4. 压缩 SVG 中的 base64 图片
# =====================================================
def compress_embedded_images_in_svg(svg_text, max_width=1400, jpeg_quality=85):
    """
    查找并压缩 SVG 中的 base64 图片。
    支持：
    data:image/png;base64,...
    data:image/jpeg;base64,...
    data:image/jpg;base64,...
    data:image/webp;base64,...
    """

    pattern = re.compile(
        r'data:image/(?P<fmt>png|jpeg|jpg|webp);base64,(?P<data>[A-Za-z0-9+/=\s]+)',
        re.IGNORECASE
    )

    count = 0
    total_before = 0
    total_after = 0

    def repl(match):
        nonlocal count, total_before, total_after

        b64_data = re.sub(r"\s+", "", match.group("data"))

        try:
            raw = base64.b64decode(b64_data)
            before_size = len(raw)

            new_data_uri, original_px, new_px, after_size = compress_image_bytes(
                raw,
                max_width=max_width,
                jpeg_quality=jpeg_quality
            )

            count += 1
            total_before += before_size
            total_after += after_size

            print(
                f"  Image {count}: "
                f"{original_px[0]}x{original_px[1]} px -> {new_px[0]}x{new_px[1]} px, "
                f"{before_size / 1024:.1f} KB -> {after_size / 1024:.1f} KB"
            )

            return new_data_uri

        except Exception as e:
            print(f"  Warning: failed to compress one embedded image: {e}")
            return match.group(0)

    new_svg_text = pattern.sub(repl, svg_text)

    return new_svg_text, count, total_before, total_after


# =====================================================
# 5. 处理单个 SVG 文件
# =====================================================
def process_svg_file(svg_path, output_dir, max_width=1400, jpeg_quality=85):
    svg_path = Path(svg_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"Processing: {svg_path.name}")

    original_file_size = svg_path.stat().st_size

    # 读取 SVG 文本
    try:
        svg_text = svg_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        svg_text = svg_path.read_text(encoding="utf-8-sig")

    new_svg_text, count, before_img, after_img = compress_embedded_images_in_svg(
        svg_text,
        max_width=max_width,
        jpeg_quality=jpeg_quality
    )

    output_path = output_dir / svg_path.name
    output_path.write_text(new_svg_text, encoding="utf-8")

    new_file_size = output_path.stat().st_size

    print(f"  Embedded images found: {count}")
    print(
        f"  Embedded image bytes: "
        f"{before_img / 1024 / 1024:.2f} MB -> {after_img / 1024 / 1024:.2f} MB"
    )
    print(
        f"  SVG file size: "
        f"{original_file_size / 1024 / 1024:.2f} MB -> {new_file_size / 1024 / 1024:.2f} MB"
    )
    print(f"  Saved to: {output_path}")

    if count == 0:
        print("  注意：没有找到 base64 内嵌图片。")
        print("  可能原因：")
        print("  1. 图片是外部链接，而不是嵌入在 SVG 里；")
        print("  2. SVG 里没有位图；")
        print("  3. 图片格式不是常见的 png/jpeg/webp data URI。")

    return output_path


# =====================================================
# 6. 主程序：批量处理文件夹里的所有 SVG
# =====================================================
def main():
    svg_dir = Path(SVG_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not svg_dir.exists():
        print(f"错误：文件夹不存在：{svg_dir}")
        return

    svg_paths = sorted(svg_dir.glob("*.svg"))

    print(f"Input folder: {svg_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Found {len(svg_paths)} SVG files.")
    print(f"MAX_IMAGE_WIDTH = {MAX_IMAGE_WIDTH}")
    print(f"JPEG_QUALITY = {JPEG_QUALITY}")

    if len(svg_paths) == 0:
        print("没有找到 SVG 文件，请检查文件夹路径。")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for svg_path in svg_paths:
        process_svg_file(
            svg_path=svg_path,
            output_dir=output_dir,
            max_width=MAX_IMAGE_WIDTH,
            jpeg_quality=JPEG_QUALITY
        )

    print("\n" + "=" * 70)
    print("Done.")
    print(f"Compressed SVGs saved to: {output_dir}")


if __name__ == "__main__":
    main()