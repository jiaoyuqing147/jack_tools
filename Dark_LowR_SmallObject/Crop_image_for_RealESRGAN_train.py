import os
import cv2

hr_dir = "D:/Jiao/dataset/CatOnlyCOCOVOC/Train_Real_ESRGAN/val/HR"
save_dir = "D:/Jiao/dataset/CatOnlyCOCOVOC/Train_Real_ESRGAN/val/HR_fixed"

os.makedirs(save_dir, exist_ok=True)

for img_name in os.listdir(hr_dir):
    img_path = os.path.join(hr_dir, img_name)
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # 裁剪高度和宽度到偶数
    new_h = h - (h % 2)
    new_w = w - (w % 2)
    cropped_img = img[:new_h, :new_w]

    save_path = os.path.join(save_dir, img_name)
    cv2.imwrite(save_path, cropped_img)

print("裁剪完成！")
