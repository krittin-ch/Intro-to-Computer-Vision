import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from scipy.ndimage import binary_dilation, binary_erosion, binary_opening, binary_closing
from skimage.filters import threshold_otsu

# Paths
input_path = 'sample-images/'
output_path = 'image-processing/filters/'

# Read and resize the image
img = cv.imread(os.path.join(input_path, 'image1.jpg'), cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (1000, 1000))

# Function to convert image to binary using Otsu's thresholding
def img_binary(img):
    threshold = threshold_otsu(img)
    binary_img = (img > threshold).astype(np.uint8) * 255
    return binary_img

# Dilation
def dilate_image(img, kernel_size=5):
    structure = np.ones((kernel_size, kernel_size))
    dilated = binary_dilation(img, structure=structure).astype(np.uint8) * 255
    return dilated

# Erosion
def erode_image(img, kernel_size=5):
    structure = np.ones((kernel_size, kernel_size))
    eroded = binary_erosion(img, structure=structure).astype(np.uint8) * 255
    return eroded

# Opening (erosion followed by dilation)
def opening_image(img, kernel_size=5):
    structure = np.ones((kernel_size, kernel_size))
    opened = binary_opening(img, structure=structure).astype(np.uint8) * 255
    return opened

# Closing (dilation followed by erosion)
def closing_image(img, kernel_size=5):
    structure = np.ones((kernel_size, kernel_size))
    closed = binary_closing(img, structure=structure).astype(np.uint8) * 255
    return closed

# Majority (local majority filter)
def majority_filter(img, size=3):
    structure = np.ones((size, size))
    majority_img = binary_closing(binary_opening(img, structure)).astype(np.uint8) * 255
    return majority_img

# Convert the image to binary
binary_img = img_binary(img)

# Apply the filters
dilated_img = dilate_image(binary_img)
eroded_img = erode_image(binary_img)
majority_img = majority_filter(binary_img)
opened_img = opening_image(binary_img)
closed_img = closing_image(binary_img)

# Plot and save the results using subplots
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
axes = axes.ravel()

titles = ['Original Binary Image', 'Dilated Image', 'Eroded Image', 
          'Majority Filtered Image', 'Opened Image', 'Closed Image']
images = [binary_img, dilated_img, eroded_img, majority_img, opened_img, closed_img]

for ax, title, image in zip(axes, titles, images):
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.suptitle('Binary Images Processing')
plt.savefig(os.path.join(output_path, 'output-binary.jpg'))
plt.show()
