import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
gray = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

if gray is None:
    raise FileNotFoundError("Image not found! Make sure test.jpg is in the same folder.")

# --- Perform linear contrast stretching ---
min_val = np.min(gray)
max_val = np.max(gray)

# Formula: (pixel - min) / (max - min) * 255
stretched = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# --- Display results ---
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(stretched, cmap="gray")
plt.title("Contrast Stretched")
plt.axis("off")

plt.tight_layout()
plt.show()
