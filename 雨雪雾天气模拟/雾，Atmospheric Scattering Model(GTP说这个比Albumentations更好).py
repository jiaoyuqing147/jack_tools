import cv2
import numpy as np
import os


def add_fog_more_realistic(img, beta=0.12, A=0.92):
    img = img.astype(np.float32) / 255.0

    # 降低饱和度
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.55
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # 降低对比度
    img = img * 0.75 + 0.12

    h, w, _ = img.shape

    # 简单伪深度：越靠上/远处雾越明显
    y = np.linspace(1, 0, h).reshape(h, 1)
    depth = np.repeat(y, w, axis=1)

    # 大气散射模型
    t = np.exp(-beta * depth * 10)[:, :, None]
    foggy = img * t + A * (1 - t)

    return np.clip(foggy * 255, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    input_path = "/home/jiaoyuqing/tt100k_2021/yolojack/images/train/165.jpg"
    output_dir = "fog_outputs"
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    foggy = add_fog_more_realistic(img, beta=0.12, A=0.92)

    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)

    cv2.imwrite(output_path, foggy)

    print(f"Saved foggy image to: {output_path}")



#下面这个版本也比较下：
import cv2
import numpy as np

def add_fog(img, beta=0.08, A=0.9):
    img = img.astype(np.float32) / 255.0
    h, w, _ = img.shape

    y = np.linspace(0, 1, h).reshape(h, 1)
    depth = np.repeat(y, w, axis=1)

    t = np.exp(-beta * depth * 10)
    t = t[:, :, None]

    foggy = img * t + A * (1 - t)
    foggy = np.clip(foggy * 255, 0, 255).astype(np.uint8)
    return foggy

img = cv2.imread("input.jpg")
foggy = add_fog(img, beta=0.12, A=0.95)
cv2.imwrite("foggy.jpg", foggy)