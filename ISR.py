from realesrgan import RealESRGAN
import cv2

# 加载 Real-ESRGAN 模型
model = RealESRGAN("weights/realesrgan-x4.pth")

# 读取低分辨率图像
image = cv2.imread(r"D:\Jiao\dataset\Jack_generate_cat\DarkResizeYolostyle\images\train2017\000000000443.jpg")

# 进行超分辨率
sr_image = model.enhance(image)

# 保存结果
cv2.imwrite("sr_cat.jpg", sr_image)

print("✅ 超分辨率完成，已保存 sr_cat.jpg")
