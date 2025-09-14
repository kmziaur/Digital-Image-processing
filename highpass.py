import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. Load the grayscale image
# ------------------------------
image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image file not found. Make sure 'test.jpg' exists in your working directory.")

# Get image size
rows, cols = image.shape
crow, ccol = rows//2 , cols//2  # center

# ------------------------------
# 2. Compute 2D Fourier Transform
# ------------------------------
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# ------------------------------
# 3. Design Ideal High-Pass Filter (IHPF)
# ------------------------------
D0 = 30  # Cutoff frequency (adjustable)
mask = np.ones((rows, cols), np.uint8)
y, x = np.ogrid[:rows, :cols]
distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
mask[distance <= D0] = 0  # Block low frequencies, pass high frequencies

# ------------------------------
# 4. Apply the filter in frequency domain
# ------------------------------
fshift_filtered = fshift * mask

# Inverse Fourier Transform to get the filtered image
f_ishift = np.fft.ifftshift(fshift_filtered)
image_filtered = np.fft.ifft2(f_ishift)
image_filtered = np.abs(image_filtered)  # Take magnitude

# ------------------------------
# 5. Display Original and Filtered Images
# ------------------------------
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(image_filtered, cmap='gray')
plt.title('High-Pass Filtered Image (IHPF)')
plt.axis('off')

plt.show()
