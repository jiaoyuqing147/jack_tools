import json
import os
from PIL import Image, ImageDraw, ImageColor, ImageFont

# === 路径修改成你的本地绝对路径 ===
# 标注目录（就是当前文件夹下的 annotations）
ANNOTATIONS_DIR = r"E:\DataSets\Mapillary\mtsd_fully_annotated_annotation\mtsd_v2_fully_annotated\annotations"

# 图片目录（你前面解压的 fully_annotated_images）
IMAGES_DIR = r"E:\DataSets\Mapillary\mtsd_fully_annotated_annotation\images"

def load_annotation(image_key):
    anno_path = os.path.join(ANNOTATIONS_DIR, f"{image_key}.json")
    with open(anno_path, "r", encoding="utf-8") as fid:
        anno = json.load(fid)
    return anno

def visualize_gt(image_key, anno, color="green", alpha=125, font=None):
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        print("Falling back to default font...")
        font = ImageFont.load_default()

    img_path = os.path.join(IMAGES_DIR, f"{image_key}.jpg")
    with Image.open(img_path) as img:
        img = img.convert("RGBA")
        img_draw = ImageDraw.Draw(img)
        rects = Image.new("RGBA", img.size)
        rects_draw = ImageDraw.Draw(rects)

        for obj in anno["objects"]:
            x1 = obj["bbox"]["xmin"]
            y1 = obj["bbox"]["ymin"]
            x2 = obj["bbox"]["xmax"]
            y2 = obj["bbox"]["ymax"]

            color_tuple = ImageColor.getrgb(color)
            if len(color_tuple) == 3:
                color_tuple = color_tuple + (alpha,)
            else:
                color_tuple[-1] = alpha

            rects_draw.rectangle((x1 + 1, y1 + 1, x2 - 1, y2 - 1), fill=color_tuple)
            img_draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill="black", width=1)

            class_name = obj["label"]
            img_draw.text((x1 + 5, y1 + 5), class_name, font=font)

        img = Image.alpha_composite(img, rects)
        return img

if __name__ == "__main__":
    # 示例图片名（你可以换成任意文件名，不带后缀）
    image_key = "Bh36Ed4HBJatMpSNnFTgTw"

    anno = load_annotation(image_key)
    vis_img = visualize_gt(image_key, anno)
    vis_img.show()
