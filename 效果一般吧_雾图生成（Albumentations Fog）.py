import cv2
import numpy as np

img = cv2.imread(
    r"E:\DataSets\tt100k_2021_paper2\tt100k_60\images\train\35.jpg"
)

fog = img.astype(np.float32)

h, w = img.shape[:2]

A = 255

for y in range(h):

    d = y / h

    t = np.exp(-3 * d)

    fog[y, :, :] = (
        fog[y, :, :] * t +
        A * (1 - t)
    )

fog = np.clip(
    fog,
    0,
    255
).astype(np.uint8)

cv2.imwrite(
    "35_fog_test.jpg",
    fog
)

print("Done")