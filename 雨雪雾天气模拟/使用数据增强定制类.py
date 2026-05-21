import cv2
import albumentations as A

# ===============================
# 1. 定义增强（2.0.8 官方参数）
# ===============================
augment = A.Compose(
    [
        # ---- 天气效果（任选其一）----
        A.OneOf(
            [
                A.RandomRain(p=1.0),
                A.RandomFog(p=1.0),
                A.RandomSnow(p=1.0),
                A.RandomShadow(p=1.0),
            ],
            p=0.3,
        ),

        # ---- 颜色增强 ----
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=15,
            p=0.3,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3,
        ),

        # ---- 模糊（任选其一）----
        A.OneOf(
            [
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
            ],
            p=0.25,
        ),

        # ---- 几何畸变 ----
        A.OneOf(
            [
                A.GridDistortion(num_steps=5, p=1.0),
                A.OpticalDistortion(distort_limit=0.05, p=1.0),
            ],
            p=0.2,
        ),

        # ---- 遮挡 ----
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            p=0.2,
        ),

        # ---- 如果你一定要 640×640 ----
        A.Resize(640, 640),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["labels"],
        min_visibility=0.2,
    ),
)

# ===============================
# 2. 读取图像 + YOLO bbox
# ===============================
img = cv2.imread("62.jpg")  # BGR
bboxes = [(0.5, 0.5, 0.2, 0.1)]  # (xc, yc, w, h)
labels = [0]

# ===============================
# 3. 执行增强
# ===============================
out = augment(image=img, bboxes=bboxes, labels=labels)

img_aug = out["image"]
bboxes_aug = out["bboxes"]
labels_aug = out["labels"]

# ===============================
# 4. 保存结果
# ===============================
cv2.imwrite("62_aug.jpg", img_aug)
print("增强后的 bbox:", bboxes_aug)#帮助调整txt标注的东西
