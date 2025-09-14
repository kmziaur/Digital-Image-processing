import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")   # or "Qt5Agg" if installed

# Load grayscale image
gray = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

# Check if image is loaded
if gray is None:
    raise FileNotFoundError("Image not found! Make sure test.jpg is in this folder.")

# Subsampling
factors = [2, 4, 8]
plt.figure(figsize=(12, 4))

# Original
plt.subplot(1, len(factors)+1, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original")
plt.axis("off")

# Subsampled versions
for i, f in enumerate(factors):
    small = gray[::f, ::f]
    plt.subplot(1, len(factors)+1, i+2)
    plt.imshow(small, cmap="gray")
    plt.title(f"Subsample {f}x")
    plt.axis("off")

plt.show()
