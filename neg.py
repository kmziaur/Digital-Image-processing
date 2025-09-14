import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
gray = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

# Create negative
negative = 255 - gray

# Display
plt.subplot(1,2,1); plt.imshow(gray, cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(negative, cmap="gray"); plt.title("Negative"); plt.axis("off")
plt.show()
