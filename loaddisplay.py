import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")   # or "Qt5Agg" if installed

# Load grayscale and color image
gray = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
color = cv2.imread("test.jpg", cv2.IMREAD_COLOR)
color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)  # convert BGR → RGB

plt.subplot(1,2,1); plt.imshow(gray, cmap="gray"); plt.title("Grayscale")
plt.subplot(1,2,2); plt.imshow(color); plt.title("Color")
plt.show()
