import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

# Paths
input_path = 'sample-images/'
output_path = 'image-processing/filters/'

# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Read and resize the image
img = cv.imread(os.path.join(input_path, 'image1.jpg'), cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (1000, 1000))

# Function to convert image to binary
def img_binary(img, threshold=128):
    _, binary_img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    return binary_img

# Dilation
def dilate_image(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv.dilate(img, kernel, iterations=1)

# Erosion
def erode_image(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv.erode(img, kernel, iterations=1)

# Majority (local majority filter)
def majority_filter(img, size=3):
    kernel = np.ones((size, size), np.uint8)
    return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

# Opening (erosion followed by dilation)
def opening_image(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

# Closing (dilation followed by erosion)
def closing_image(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

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
