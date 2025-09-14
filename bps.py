import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load grayscale image
gray = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

if gray is None:
    raise FileNotFoundError("Image not found! Make sure test.jpg is in the same folder.")

plt.figure(figsize=(12,6))

# Extract and display 8 bit-planes
for i in range(7, -1, -1):   # MSB=7 → LSB=0
    bit_plane = (gray >> i) & 1    # shift & mask
    plt.subplot(2, 4, 8-i)         # arrange in order MSB→LSB
    plt.imshow(bit_plane * 255, cmap="gray")
    plt.title(f"Bit Plane {i}")
    plt.axis("off")

plt.tight_layout()
plt.show()
