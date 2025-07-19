from PIL import Image
import os

img_dir = r'F:\jack_dataset\EMTD\yolo66\images\train'
for img_name in os.listdir(img_dir):
    path = os.path.join(img_dir, img_name)
    try:
        img = Image.open(path)
        img.verify()  # 仅检查，不加载
    except Exception as e:
        print(f"{img_name} is corrupt: {e}")
