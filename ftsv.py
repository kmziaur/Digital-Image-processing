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
# 2. Compute 2D Fourier Transform
# ------------------------------
# Compute 2D Fourier Transform
f = np.fft.fft2(image)

# Shift zero-frequency component to the center
fshift = np.fft.fftshift(f)

# Compute magnitude spectrum using logarithmic scaling
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Add 1 to avoid log(0)

# ------------------------------
# 3. Display Original and Spectrum
# ------------------------------
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum (Log Scale)')
plt.axis('off')

plt.show()
