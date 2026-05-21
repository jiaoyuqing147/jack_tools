# aug_albu_custom.py
# pip install albumentations opencv-python

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import albumentations as A


@dataclass
class AugConfig:
    # --- 全局开关/概率 ---
    p_weather: float = 0.25
    p_dropout: float = 0.20
    p_color: float = 0.35
    p_blur: float = 0.25
    p_distort: float = 0.20

    # --- 输入尺寸（如果你要统一到 640）---
    enable_resize: bool = False
    out_size: int = 640  # enable_resize=True 时生效

    # --- CoarseDropout 参数 ---
    # 常用于模拟遮挡/脏污；小目标任务建议不要太狠
    max_holes: int = 8
    max_height: int = 32
    max_width: int = 32
    min_holes: int = 1
    min_height: int = 8
    min_width: int = 8

    # --- 颜色增强参数 ---
    hue_shift_limit: int = 10
    sat_shift_limit: int = 20
    val_shift_limit: int = 15

    # --- 亮度对比度 ---
    brightness_limit: float = 0.15
    contrast_limit: float = 0.15

    # --- 模糊 ---
    motion_blur_limit: int = 9
    gaussian_blur_limit: Tuple[int, int] = (3, 7)  # 只能奇数核，Albumentations 会处理
    median_blur_limit: int = 7
    random_blur_limit: Tuple[int, int] = (3, 9)    # Blur 的核大小范围（奇数）

    # --- 畸变 ---
    grid_distort_num_steps: int = 5
    optical_distort_limit: float = 0.05
    optical_shift_limit: float = 0.05

    # --- 天气效果（强度尽量保守，尤其小目标）---
    fog_coef_lower: float = 0.05
    fog_coef_upper: float = 0.20
    fog_alpha_coef: float = 0.08

    rain_slant_lower: int = -10
    rain_slant_upper: int = 10
    rain_drop_length: int = 12
    rain_drop_width: int = 1
    rain_brightness_coefficient: float = 0.95

    snow_brightness_coeff: float = 1.05
    snow_snow_point_lower: float = 0.10
    snow_snow_point_upper: float = 0.25

    shadow_num_shadows_lower: int = 1
    shadow_num_shadows_upper: int = 2
    shadow_shadow_dimension: int = 5

    sunflare_src_radius: int = 120


def build_train_augmentations_yolo(
    cfg: AugConfig,
    *,
    bbox_label_fields: Tuple[str, ...] = ("class_labels",),
    min_visibility: float = 0.20,
    clip_bboxes: bool = True,
) -> A.Compose:
    """
    训练增强：支持 YOLO bbox (x_center,y_center,w,h) 归一化格式。
    你传入 bboxes=[(xc,yc,w,h), ...], class_labels=[cls_id,...] 即可。
    """

    weather = A.OneOf(
        [
            A.RandomRain(
                slant_lower=cfg.rain_slant_lower,
                slant_upper=cfg.rain_slant_upper,
                drop_length=cfg.rain_drop_length,
                drop_width=cfg.rain_drop_width,
                brightness_coefficient=cfg.rain_brightness_coefficient,
                p=1.0,
            ),
            A.RandomFog(
                fog_coef_lower=cfg.fog_coef_lower,
                fog_coef_upper=cfg.fog_coef_upper,
                alpha_coef=cfg.fog_alpha_coef,
                p=1.0,
            ),
            A.RandomSnow(
                brightness_coeff=cfg.snow_brightness_coeff,
                snow_point_lower=cfg.snow_snow_point_lower,
                snow_point_upper=cfg.snow_snow_point_upper,
                p=1.0,
            ),
            A.RandomShadow(
                num_shadows_lower=cfg.shadow_num_shadows_lower,
                num_shadows_upper=cfg.shadow_num_shadows_upper,
                shadow_dimension=cfg.shadow_shadow_dimension,
                p=1.0,
            ),
            A.RandomSunFlare(
                src_radius=cfg.sunflare_src_radius,
                p=1.0,
            ),
        ],
        p=cfg.p_weather,
    )

    color = A.OneOf(
        [
            A.HueSaturationValue(
                hue_shift_limit=cfg.hue_shift_limit,
                sat_shift_limit=cfg.sat_shift_limit,
                val_shift_limit=cfg.val_shift_limit,
                p=1.0,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_limit,
                contrast_limit=cfg.contrast_limit,
                p=1.0,
            ),
        ],
        p=cfg.p_color,
    )

    blur = A.OneOf(
        [
            A.MotionBlur(blur_limit=cfg.motion_blur_limit, p=1.0),
            A.GaussianBlur(blur_limit=cfg.gaussian_blur_limit, p=1.0),
            A.MedianBlur(blur_limit=cfg.median_blur_limit, p=1.0),
            A.Blur(blur_limit=cfg.random_blur_limit, p=1.0),  # 核大小随机
        ],
        p=cfg.p_blur,
    )

    distort = A.OneOf(
        [
            A.GridDistortion(num_steps=cfg.grid_distort_num_steps, p=1.0),
            A.OpticalDistortion(
                distort_limit=cfg.optical_distort_limit,
                shift_limit=cfg.optical_shift_limit,
                p=1.0,
            ),
        ],
        p=cfg.p_distort,
    )

    dropout = A.CoarseDropout(
        max_holes=cfg.max_holes,
        max_height=cfg.max_height,
        max_width=cfg.max_width,
        min_holes=cfg.min_holes,
        min_height=cfg.min_height,
        min_width=cfg.min_width,
        p=cfg.p_dropout,
    )

    ops: List[A.BasicTransform] = []

    if cfg.enable_resize:
        ops.append(A.Resize(cfg.out_size, cfg.out_size, p=1.0))

    # 组合（注意：畸变/模糊/颜色/天气/遮挡都在这）
    ops.extend([weather, color, blur, distort, dropout])

    return A.Compose(
        ops,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=list(bbox_label_fields),
            min_visibility=min_visibility,
            clip=clip_bboxes,
        ),
    )


def build_val_augmentations_yolo(cfg: AugConfig) -> A.Compose:
    """验证/测试：通常只做 Resize（可选）"""
    ops: List[A.BasicTransform] = []
    if cfg.enable_resize:
        ops.append(A.Resize(cfg.out_size, cfg.out_size, p=1.0))
    return A.Compose(
        ops,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def apply_aug_yolo(
    aug: A.Compose,
    image,
    bboxes: List[Tuple[float, float, float, float]],
    class_labels: List[int],
) -> Dict[str, Any]:
    """
    统一调用入口：
    - image: numpy.ndarray (H,W,3) BGR/RGB 都行（但你后续保持一致）
    - bboxes: YOLO 归一化 bbox 列表
    - class_labels: bbox 对应类别
    """
    return aug(image=image, bboxes=bboxes, class_labels=class_labels)
