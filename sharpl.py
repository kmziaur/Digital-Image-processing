import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. Load the image in grayscale
# ------------------------------
image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image file not found. Make sure 'test.jpg' exists in your working directory.")

# ------------------------------
# 2. Apply Laplacian Filter
# ------------------------------
# Use 64-bit float to handle negative values
laplacian = cv2.Laplacian(image, cv2.CV_64F)

# Sharpened image = original + Laplacian
sharpened = cv2.convertScaleAbs(image + laplacian)

# ------------------------------
# 3. Display Original and Sharpened Images
# ------------------------------
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(sharpened, cmap='gray')
plt.title('Sharpened Image with Laplacian')
plt.axis('off')

plt.show()
