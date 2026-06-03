import torch
import cv2
import numpy as np
from pathlib import Path

# =====================================================
# 输入图片
# =====================================================

IMG_PATH = Path(
    r"E:\DataSets\tt100k_2021_paper2\tt100k_60\images\train\35.jpg"
)

SAVE_PATH = Path(
    "35_fog_midas.jpg"
)

# =====================================================
# 参数
# =====================================================

FOG_STRENGTH = 0.4

FOG_COLOR = np.array(
    [235, 235, 235],
    dtype=np.uint8
)

MODEL = "DPT_Hybrid"

# =====================================================
# 加载 MiDaS
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

midas = torch.hub.load(
    "intel-isl/MiDaS",
    MODEL
)

midas.to(device)
midas.eval()

midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)

transform = midas_transforms.dpt_transform

# =====================================================
# 读取图片
# =====================================================

img_bgr = cv2.imread(str(IMG_PATH))

img_rgb = cv2.cvtColor(
    img_bgr,
    cv2.COLOR_BGR2RGB
)

# =====================================================
# 深度估计
# =====================================================

input_batch = transform(
    img_rgb
).to(device)

with torch.no_grad():

    prediction = midas(
        input_batch
    )

depth = (
    prediction
    .squeeze()
    .cpu()
    .numpy()
)

# =====================================================
# 深度归一化
# =====================================================

depth = cv2.normalize(
    depth,
    None,
    0,
    1,
    cv2.NORM_MINMAX
)

print("before resize:", depth.shape)

depth = cv2.resize(
    depth,
    (img_rgb.shape[1], img_rgb.shape[0]),
    interpolation=cv2.INTER_CUBIC
)

print("after resize:", depth.shape)
print("image:", img_rgb.shape)

# =====================================================
# 关键修改
# =====================================================
# MiDaS:
# 白色 = 远处
# 黑色 = 近处
#
# 所以直接使用 depth
# 不要反转
# =====================================================

fog_intensity = np.power(depth, 3) * FOG_STRENGTH

fog_intensity = np.clip(
    fog_intensity,
    0,
    1
)

# =====================================================
# 生成雾图
# =====================================================

fog_layer = np.ones_like(
    img_rgb,
    dtype=np.float32
) * FOG_COLOR

foggy = (
    img_rgb *
    (1 - fog_intensity[:, :, np.newaxis])
    +
    fog_layer *
    fog_intensity[:, :, np.newaxis]
)

foggy = np.clip(
    foggy,
    0,
    255
).astype(np.uint8)

# =====================================================
# 保存
# =====================================================

foggy_bgr = cv2.cvtColor(
    foggy,
    cv2.COLOR_RGB2BGR
)

cv2.imwrite(
    str(SAVE_PATH),
    foggy_bgr
)

print()
print("Saved:")
print(SAVE_PATH.resolve())
depth_vis = (depth * 255).astype(np.uint8)

cv2.imwrite(
    "depth.jpg",
    depth_vis
)