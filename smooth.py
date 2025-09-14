import cv2
import matplotlib.pyplot as plt

# Load grayscale image
gray = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

if gray is None:
    raise FileNotFoundError("Image not found! Make sure test.jpg is in the same folder.")

# --- Apply averaging filters ---
avg3 = cv2.blur(gray, (3,3))   # 3x3 kernel
avg5 = cv2.blur(gray, (5,5))   # 5x5 kernel

# --- Display results ---
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(gray, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(avg3, cmap="gray")
plt.title("3x3 Averaging")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(avg5, cmap="gray")
plt.title("5x5 Averaging")
plt.axis("off")

plt.tight_layout()
plt.show()
