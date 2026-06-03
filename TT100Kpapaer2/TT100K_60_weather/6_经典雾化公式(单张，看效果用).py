"""
Atmospheric Scattering Model for Fog Generation
==============================================

本代码采用经典大气散射模型（Atmospheric Scattering Model）
生成雾天图像。该模型广泛应用于：

    - Foggy Cityscapes
    - RESIDE
    - AOD-Net
    - Dehazing/Fog Simulation Research

雾化公式：

    I(x) = J(x) * t(x) + A * (1 - t(x))

其中：

    J(x) : 原始清晰图像（Clear Image）
    I(x) : 雾化后图像（Foggy Image）
    A    : 大气光（Atmospheric Light）
    t(x) : 透射率（Transmission Map）

透射率定义为：

    t(x) = exp(-β * d(x))

其中：

    β    : 雾浓度系数（Fog Density）
    d(x) : 场景深度（Scene Depth）

本实现针对交通场景（TT100K、GTSDB）进行了简化：

    1. 不进行真实深度估计；
    2. 利用道路场景先验，
       认为靠近地平线区域距离更远；
    3. 根据图像垂直位置构造距离图；
    4. 距离越远，雾浓度越大。

特点：

    - 生成速度快
    - 无需额外模型
    - 保持目标标注不变
    - 适用于交通标志检测数据集构建

作者：
    Jiao Yuqing

日期：
    2026
"""

import cv2
import numpy as np

img = cv2.imread(
    r"E:\DataSets\tt100k_2021_paper2\tt100k_60\images\train\35.jpg"
)

img = img.astype(np.float32)

h, w = img.shape[:2]

# ======================
# 参数
# ======================

A = 245       # 大气光

beta = 2.0    # 雾浓度

# ======================
# 构造距离图
# ======================

distance = np.zeros(
    (h, w),
    dtype=np.float32
)

for y in range(h):

    # 地平线附近最远

    d = 1.0 - (y / h)

    distance[y, :] = d

# ======================
# 大气散射模型
# ======================

t = np.exp(
    -beta * distance
)

t = np.expand_dims(
    t,
    axis=2
)

foggy = (
    img * t
    +
    A * (1 - t)
)

foggy = np.clip(
    foggy,
    0,
    255
).astype(np.uint8)

cv2.imwrite(
    "35_fog_cityscapes.jpg",
    foggy
)

print("Done")