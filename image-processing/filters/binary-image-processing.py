import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

# Paths
input_path = 'sample-images/'
output_path = 'image-processing/filters/'

# Read and resize the image
img = cv.imread(os.path.join(input_path, 'image1.jpg'), cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (1000, 1000))

# Function to convert image to binary
def img_binary(img, threshold=128):
    binary_img = (img > threshold).astype(np.uint8) * 255
    return binary_img

# Function to apply a convolution with a kernel
def apply_convolution(img, kernel):
    return cv.filter2D(img, -1, kernel)

# Dilation
def dilate_image(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    output = apply_convolution(img, kernel)
    output = (output > 0).astype(np.uint8) * 255
    return output

# Erosion
def erode_image(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    output = apply_convolution(img, kernel)
    output = (output == kernel_size * kernel_size * 255).astype(np.uint8) * 255
    return output

# Majority (local majority filter)
def majority_filter(img, size=3):
    padded_img = np.pad(img, ((size // 2, size // 2), (size // 2, size // 2)), mode='constant', constant_values=0)
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            local_patch = padded_img[i:i + size, j:j + size]
            output[i, j] = 255 if np.sum(local_patch) > (size * size * 255) / 2 else 0
    return output

# Opening (erosion followed by dilation)
def opening_image(img, kernel_size=5):
    eroded = erode_image(img, kernel_size)
    opened = dilate_image(eroded, kernel_size)
    return opened

# Closing (dilation followed by erosion)
def closing_image(img, kernel_size=5):
    dilated = dilate_image(img, kernel_size)
    closed = erode_image(dilated, kernel_size)
    return closed

# Convert the image to binary
binary_img = img_binary(img)

# Apply the filters
dilated_img = dilate_image(binary_img)
eroded_img = erode_image(binary_img)
majority_img = majority_filter(binary_img)
opened_img = opening_image(binary_img)
closed_img = closing_image(binary_img)

# Plot and save the results using subplots
fig, axes = plt.subplots(2, 3, figsize=(5, 5))
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
