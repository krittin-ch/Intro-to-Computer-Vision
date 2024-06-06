import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# Input and output paths
input_path = 'sample-images/'
output_path = 'image-processing/filters/'

# Read the RGB image
img_rgb = cv.imread(os.path.join(input_path, 'image3.jpg'))
img_rgb = cv.resize(img_rgb, (100, 100))

# Define the Bayer pattern arrangement (e.g., RGGB)

def generate_bayer_pattern(n):
    """
    Generate an n x n Bayer mosaic pattern.

    Parameters:
        n (int): Size of the Bayer mosaic pattern.

    Returns:
        numpy.ndarray: Bayer mosaic pattern.
    """
    bayer_pattern = np.zeros((n, n), dtype=np.uint8)
    bayer_pattern[::2, ::2] = 0  # R
    bayer_pattern[::2, 1::2] = 1  # G1
    bayer_pattern[1::2, ::2] = 1  # G1
    bayer_pattern[1::2, 1::2] = 2  # B
    return bayer_pattern

n = 10
bayer_pattern = generate_bayer_pattern(n)

# Create an empty Bayer mosaic image
h, w, _ = img_rgb.shape
bayer_mosaic = np.zeros(img_rgb.shape, dtype=np.uint8)

# Fill in the Bayer pattern
for y in range(h):
    for x in range(w):
        bayer_mosaic[y, x, :] = np.clip(img_rgb[y, x, :]*bayer_pattern[y % n, x % n], 0, 255)

# Display the original RGB image and the Bayer mosaic pattern
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB))
plt.title('Original RGB Image')

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(bayer_mosaic, cv.COLOR_BGR2RGB))
plt.title('Bayer Mosaic Pattern')

plt.show()
