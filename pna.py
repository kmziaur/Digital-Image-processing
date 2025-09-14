import cv2
import numpy as np

# Load grayscale image
gray = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

if gray is None:
    raise FileNotFoundError("Image not found! Make sure test.jpg is in the same folder.")

rows, cols = gray.shape

# Choose a pixel (for example, center of image)
row, col = rows // 2, cols // 2
pixel_val = gray[row, col]

# --- 4-neighbors (top, bottom, left, right) ---
neighbors_4 = []
if row > 0: neighbors_4.append(("Top", gray[row-1, col]))
if row < rows-1: neighbors_4.append(("Bottom", gray[row+1, col]))
if col > 0: neighbors_4.append(("Left", gray[row, col-1]))
if col < cols-1: neighbors_4.append(("Right", gray[row, col+1]))

# --- 8-neighbors (include diagonals) ---
neighbors_8 = []
for i in [-1, 0, 1]:
    for j in [-1, 0, 1]:
        if i == 0 and j == 0:
            continue  # skip the pixel itself
        nr, nc = row+i, col+j
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors_8.append(((nr, nc), gray[nr, nc]))

# --- Print results ---
print(f"Pixel chosen at ({row},{col}) = {pixel_val}")
print("\n4-Neighbors:")
for name, val in neighbors_4:
    print(f"{name}: {val}")

print("\n8-Neighbors:")
for pos, val in neighbors_8:
    print(f"Pixel {pos} = {val}")
