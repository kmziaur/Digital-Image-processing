import cv2
import matplotlib.pyplot as plt

# Load grayscale image
gray = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

if gray is None:
    raise FileNotFoundError("Image not found! Make sure test.jpg is in the same folder.")

# --- Original histogram ---
hist_orig = cv2.calcHist([gray], [0], None, [256], [0,256])

# --- Histogram equalization ---
equalized = cv2.equalizeHist(gray)
hist_eq = cv2.calcHist([equalized], [0], None, [256], [0,256])

# --- Display results ---
plt.figure(figsize=(10,6))

# Original image + histogram
plt.subplot(2,2,1)
plt.imshow(gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.plot(hist_orig, color='black')
plt.title("Original Histogram")
plt.xlim([0,256])

# Equalized image + histogram
plt.subplot(2,2,3)
plt.imshow(equalized, cmap="gray")
plt.title("Equalized Image")
plt.axis("off")

plt.subplot(2,2,4)
plt.plot(hist_eq, color='black')
plt.title("Equalized Histogram")
plt.xlim([0,256])

plt.tight_layout()
plt.show()
