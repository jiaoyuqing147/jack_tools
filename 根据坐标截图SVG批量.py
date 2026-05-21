import re
import base64
import copy
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

# ========= 输入文件夹 =========
svg_dir = Path(r"C:\Users\JACKJIAO\Desktop\00239Heatmap_resize1280")



x1, y1 = 737, 636
x2, y2 = 858, 757


# ========= 裁剪坐标 =========
# 备用坐标（tt100k的第二个）
# x1, y1 = 1297, 922
# x2, y2 = 1343, 968

# 备用坐标（tt100k的第一个）
# x1, y1 = 317, 1006
# x2, y2 = 355, 1044

if x2 <= x1 or y2 <= y1:
    raise ValueError("裁剪区域无效：右下角必须大于左上角")

crop_w = x2 - x1
crop_h = y2 - y1

# ========= 检查文件夹 =========
if not svg_dir.exists():
    raise FileNotFoundError(f"文件夹不存在: {svg_dir}")

svg_files = list(svg_dir.glob("*.svg"))
if not svg_files:
    raise FileNotFoundError(f"文件夹中没有找到 SVG 文件: {svg_dir}")


def get_local_tag(tag):
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def strip_unit(val):
    if val is None:
        return None
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)", str(val))
    return float(m.group(1)) if m else None


def parse_svg_size(root):
    viewbox = root.get("viewBox")
    if viewbox:
        vals = list(map(float, viewbox.replace(",", " ").split()))
        if len(vals) == 4:
            return vals
    w = strip_unit(root.get("width"))
    h = strip_unit(root.get("height"))
    if w is not None and h is not None:
        return [0.0, 0.0, w, h]
    return None


def find_image_element(root):
    for elem in root.iter():
        if get_local_tag(elem.tag) == "image":
            return elem
    return None


def get_image_href(image_elem):
    href = image_elem.get("href")
    if href:
        return href
    for k, v in image_elem.attrib.items():
        if k.endswith("}href") or k == "xlink:href":
            return v
    return None


def in_crop_rect(x, y, w, h, crop_x, crop_y, crop_x2, crop_y2):
    """矩形与裁剪区域是否有交集"""
    if x is None or y is None or w is None or h is None:
        return False
    return not (x + w <= crop_x or x >= crop_x2 or y + h <= crop_y or y >= crop_y2)


def point_in_crop(x, y, crop_x, crop_y, crop_x2, crop_y2):
    if x is None or y is None:
        return False
    return crop_x <= x <= crop_x2 and crop_y <= y <= crop_y2


for svg_path in svg_files:
    try:
        text = svg_path.read_text(encoding="utf-8")
        text_clean = re.sub(r'<!DOCTYPE[^>]*>', '', text, flags=re.IGNORECASE)
        root = ET.fromstring(text_clean)

        # 命名空间
        if root.tag.startswith("{"):
            ns_uri = root.tag.split("}")[0][1:]
            svg_ns = f"{{{ns_uri}}}"
            ET.register_namespace("", ns_uri)
        else:
            ns_uri = "http://www.w3.org/2000/svg"
            svg_ns = ""

        size_info = parse_svg_size(root)
        if not size_info:
            print(f"⚠ 跳过 {svg_path.name}：无法识别 SVG 尺寸")
            continue

        vb_x, vb_y, vb_w, vb_h = size_info

        crop_x = max(vb_x, x1)
        crop_y = max(vb_y, y1)
        crop_x2 = min(vb_x + vb_w, x2)
        crop_y2 = min(vb_y + vb_h, y2)

        real_crop_w = crop_x2 - crop_x
        real_crop_h = crop_y2 - crop_y

        if real_crop_w <= 0 or real_crop_h <= 0:
            print(
                f"⚠ 跳过 {svg_path.name}：裁剪区域超出图像范围。\n"
                f"   原始范围: ({vb_x}, {vb_y}, {vb_x + vb_w}, {vb_y + vb_h})\n"
                f"   请求范围: ({x1}, {y1}, {x2}, {y2})"
            )
            continue

        # ========= 找到底图 image，并裁掉 =========
        image_elem = find_image_element(root)
        if image_elem is None:
            print(f"⚠ 跳过 {svg_path.name}：未找到 <image> 元素")
            continue

        href = get_image_href(image_elem)
        if not href:
            print(f"⚠ 跳过 {svg_path.name}：<image> 没有 href/xlink:href")
            continue

        m = re.match(r"data:image/([a-zA-Z0-9+]+);base64,(.+)", href, re.DOTALL)
        if not m:
            print(f"⚠ 跳过 {svg_path.name}：<image> 不是 base64 内嵌图片")
            continue

        img_b64 = m.group(2).strip()
        img_data = base64.b64decode(img_b64)
        img = Image.open(BytesIO(img_data))

        img_w, img_h = img.size

        img_x = strip_unit(image_elem.get("x")) or 0.0
        img_y = strip_unit(image_elem.get("y")) or 0.0
        disp_w = strip_unit(image_elem.get("width")) or img_w
        disp_h = strip_unit(image_elem.get("height")) or img_h

        scale_x = img_w / disp_w
        scale_y = img_h / disp_h

        px1 = int(round((crop_x - img_x) * scale_x))
        py1 = int(round((crop_y - img_y) * scale_y))
        px2 = int(round((crop_x2 - img_x) * scale_x))
        py2 = int(round((crop_y2 - img_y) * scale_y))

        px1 = max(0, min(img_w, px1))
        py1 = max(0, min(img_h, py1))
        px2 = max(0, min(img_w, px2))
        py2 = max(0, min(img_h, py2))

        if px2 <= px1 or py2 <= py1:
            print(f"⚠ 跳过 {svg_path.name}：映射到图片后的裁剪区域无效")
            continue

        cropped = img.crop((px1, py1, px2, py2))

        buffer = BytesIO()
        cropped.save(buffer, format="PNG")
        new_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # ========= 新建 SVG =========
        new_root = ET.Element(f"{svg_ns}svg")
        new_root.set("version", root.get("version", "1.1"))
        new_root.set("width", f"{real_crop_w:.4f}")
        new_root.set("height", f"{real_crop_h:.4f}")
        new_root.set("viewBox", f"0 0 {real_crop_w:.4f} {real_crop_h:.4f}")

        for k, v in root.attrib.items():
            if k not in {"width", "height", "viewBox", "version"}:
                new_root.set(k, v)

        # 先放背景图
        new_image = ET.SubElement(new_root, f"{svg_ns}image")
        new_image.set("x", "0")
        new_image.set("y", "0")
        new_image.set("width", f"{real_crop_w:.4f}")
        new_image.set("height", f"{real_crop_h:.4f}")
        new_image.set("preserveAspectRatio", "none")
        new_image.set("href", f"data:image/png;base64,{new_b64}")

        # 新 defs
        new_defs = ET.SubElement(new_root, f"{svg_ns}defs")

        # ========= 第一遍：收集要保留的 clipPath id =========
        needed_clip_ids = set()
        elements_to_add = []

        for child in list(root):
            tag = get_local_tag(child.tag)

            if tag in ("defs", "image"):
                continue

            # 保留 rect（检测框、标签背景）
            if tag == "rect":
                x = strip_unit(child.get("x"))
                y = strip_unit(child.get("y"))
                w = strip_unit(child.get("width"))
                h = strip_unit(child.get("height"))
                if in_crop_rect(x, y, w, h, crop_x, crop_y, crop_x2, crop_y2):
                    elements_to_add.append(child)

            # 保留 text
            elif tag == "text":
                x = strip_unit(child.get("x"))
                y = strip_unit(child.get("y"))
                if point_in_crop(x, y, crop_x, crop_y, crop_x2, crop_y2):
                    cp = child.get("clip-path")
                    if cp:
                        m_id = re.match(r"url\(#([^)]+)\)", cp)
                        if m_id:
                            needed_clip_ids.add(m_id.group(1))
                    elements_to_add.append(child)

            # 保留 clipPath（后面按 needed_clip_ids 再筛）
            elif tag == "clipPath":
                pass

        # ========= 第二遍：从原文件复制需要的 clipPath =========
        for child in list(root):
            if get_local_tag(child.tag) == "clipPath":
                cid = child.get("id")
                if cid in needed_clip_ids:
                    new_cp = copy.deepcopy(child)

                    # 平移 clipPath 里面的 rect
                    for sub in new_cp.iter():
                        if get_local_tag(sub.tag) == "rect":
                            x = strip_unit(sub.get("x"))
                            y = strip_unit(sub.get("y"))
                            if x is not None:
                                sub.set("x", str(x - crop_x))
                            if y is not None:
                                sub.set("y", str(y - crop_y))

                    new_defs.append(new_cp)

        # ========= 第三遍：复制 rect/text，并平移 =========
        for child in elements_to_add:
            new_elem = copy.deepcopy(child)
            tag = get_local_tag(new_elem.tag)

            if tag == "rect":
                x = strip_unit(new_elem.get("x"))
                y = strip_unit(new_elem.get("y"))
                w = strip_unit(new_elem.get("width"))
                h = strip_unit(new_elem.get("height"))

                # 这里不做几何截断，只做平移；只要相交就保留
                if x is not None:
                    new_elem.set("x", str(x - crop_x))
                if y is not None:
                    new_elem.set("y", str(y - crop_y))

            elif tag == "text":
                x = strip_unit(new_elem.get("x"))
                y = strip_unit(new_elem.get("y"))
                if x is not None:
                    new_elem.set("x", str(x - crop_x))
                if y is not None:
                    new_elem.set("y", str(y - crop_y))

            new_root.append(new_elem)

        output_path = svg_path.parent / f"{svg_path.stem}_crop.svg"
        ET.ElementTree(new_root).write(output_path, encoding="utf-8", xml_declaration=True)

        old_size_mb = svg_path.stat().st_size / (1024 * 1024)
        new_size_mb = output_path.stat().st_size / (1024 * 1024)

        print(f"✅ 已完成: {svg_path.name}")
        print(f"   输出文件: {output_path.name}")
        print(f"   SVG裁剪区域: ({crop_x}, {crop_y}) -> ({crop_x2}, {crop_y2})")
        print(f"   图片像素裁剪: ({px1}, {py1}) -> ({px2}, {py2})")
        print(f"   文件大小: {old_size_mb:.2f} MB -> {new_size_mb:.2f} MB")

    except Exception as e:
        print(f"❌ 处理失败: {svg_path.name}")
        print(f"   原因: {e}")

print("\n🎉 全部 SVG 处理完成")