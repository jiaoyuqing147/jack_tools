import albumentations as A
import cv2, os

# 定义增强效果（概率1.0表示每张图片都会被增强）
transform = A.OneOf([
    A.RandomRain(brightness_coefficient=0.95, blur_value=2, rain_type='drizzle', p=1.0),
    A.RandomFog(alpha_coef=0.05, p=1.0),
    A.RandomSnow(brightness_coeff=1.2, p=1.0)
], p=1.0)

src = "images"
dst = "images_weather"
os.makedirs(dst, exist_ok=True)

for fname in os.listdir(src):
    fpath = os.path.join(src, fname)
    img = cv2.imread(fpath)
    if img is None:
        continue
    aug = transform(image=img)["image"]
    cv2.imwrite(os.path.join(dst, fname), aug)
