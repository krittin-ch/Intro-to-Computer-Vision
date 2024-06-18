import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv

# Paths
input_path = 'sample-images/'
output_path = 'image-processing/excercises/'

# Read and resize the image
img = cv.imread(os.path.join(input_path, 'view.jpg'))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

l = 1000
img = cv.resize(img, (l, l))
img = np.array(img)  

# Function to compute histogram
def compute_histogram(image):
    # Initialize histogram arrays
    histogram = np.zeros((3, 256), dtype=int)
    
    # Compute histogram for each channel
    for i in range(3):  # Loop over R, G, B channels
        channel_values, channel_counts = np.unique(image[:, :, i], return_counts=True)
        histogram[i, channel_values] = channel_counts
    
    return histogram

# Function to perform histogram equalization
def histogram_equalization(image):
    # Compute histogram
    histogram = compute_histogram(image)
    
    # Compute cumulative distribution function (CDF)
    cdf = histogram.cumsum(axis=1)
    
    # Normalize CDF to scale it to [0, 255]
    cdf_normalized = (cdf - cdf.min(axis=1, keepdims=True)) * 255 / (cdf.max(axis=1, keepdims=True) - cdf.min(axis=1, keepdims=True))
    
    # Ensure cdf_normalized has exactly 256 unique values
    cdf_flat = np.zeros(256)
    for i in range(3):  # Loop over R, G, B channels
        unique_values, idx = np.unique(cdf_normalized[i], return_index=True)
        cdf_flat[unique_values.astype(int)] = cdf_normalized[i, idx]
    
    # Reshape image and cdf_flat to 1D arrays
    image_flat = image.flatten()
    
    # Interpolate CDF to get the equalized image
    equalized_image = np.interp(image_flat, np.arange(0, 256), cdf_flat).reshape(image.shape).astype(np.uint8)
    
    return equalized_image

# Perform histogram equalization
equalized_img = histogram_equalization(img)

# Plotting the results
fig, ax = plt.subplots(2, 2, figsize=(15, 15))

# Display original image
ax[0, 0].imshow(img)
ax[0, 0].set_title("Original Image")

# Plot histogram of original image
for i, color in enumerate(['r', 'g', 'b']):
    ax[1, 0].plot(np.arange(256), compute_histogram(img)[i], color=color)
ax[1, 0].set_title("Color Histogram (Original)")
ax[1, 0].set_xlabel("Color value")
ax[1, 0].set_ylabel("Pixel count")

# Display equalized image
ax[0, 1].imshow(equalized_img.astype(np.uint8))
ax[0, 1].set_title("Equalized Image")

# Plot histogram of equalized image
for i, color in enumerate(['r', 'g', 'b']):
    ax[1, 1].plot(np.arange(256), compute_histogram(equalized_img.astype(np.uint8))[i], color=color)
ax[1, 1].set_title("Color Histogram (Equalized)")
ax[1, 1].set_xlabel("Color value")
ax[1, 1].set_ylabel("Pixel count")


plt.savefig(os.path.join(output_path, 'output-histogram-equalization-1.jpg'))


plt.tight_layout()
plt.show()
