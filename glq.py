import numpy as np
import matplotlib.pyplot as plt
import cv2

gray = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

levels = [2, 4, 8, 16, 256]
plt.figure(figsize=(12,6))

for i, L in enumerate(levels):
    quantized = np.floor(gray / (256/L)) * (256/L)
    quantized = quantized.astype(np.uint8)
    plt.subplot(1, len(levels), i+1)
    plt.imshow(quantized, cmap="gray")
    plt.title(f"{L} levels")
    plt.axis("off")

plt.show()
