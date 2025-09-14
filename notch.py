import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. Load the grayscale image
# ------------------------------
image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image file not found. Make sure 'test.jpg' exists in your working directory.")

rows, cols = image.shape

# ------------------------------
# 2. Add synthetic periodic noise
# ------------------------------
# Create sinusoidal noise
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)
frequency = 0.1  # Adjust frequency of sinusoidal noise
sin_noise = 50 * np.sin(2 * np.pi * frequency * X)  # Horizontal stripes

# Add noise to the image
noisy_image = cv2.add(image.astype(np.float32), sin_noise.astype(np.float32))
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# ------------------------------
# 3. Fourier Transform of noisy image
# ------------------------------
f = np.fft.fft2(noisy_image)
fshift = np.fft.fftshift(f)

# ------------------------------
# 4. Design a Notch Filter to remove noise
# ------------------------------
# Create a mask with ones everywhere
mask = np.ones((rows, cols), np.uint8)

# Coordinates of the noise frequencies (manually estimated from spectrum)
crow, ccol = rows//2, cols//2
# Example: block two symmetric noise peaks
D = 5  # Notch radius
# Horizontal noise peak coordinates
coords = [(crow, ccol + int(frequency*cols)), (crow, ccol - int(frequency*cols))]

for r, c in coords:
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - c)**2 + (y - r)**2)
    mask[distance <= D] = 0  # Block the noise frequency

# ------------------------------
# 5. Apply Notch Filter
# ------------------------------
fshift_filtered = fshift * mask
f_ishift = np.fft.ifftshift(fshift_filtered)
image_filtered = np.fft.ifft2(f_ishift)
image_filtered = np.abs(image_filtered)
image_filtered = np.clip(image_filtered, 0, 255).astype(np.uint8)

# ------------------------------
# 6. Display Images
# ------------------------------
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Image with Periodic Noise')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(image_filtered, cmap='gray')
plt.title('Filtered Image (Notch Filter)')
plt.axis('off')

plt.show()
