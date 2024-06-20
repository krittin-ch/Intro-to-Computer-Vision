import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.signal
import os

input_path = 'sample-images/'
output_path = 'image-processing/filters/'

# Read and resize the image
img = cv.imread(os.path.join(input_path, 'image1.jpg'), cv.IMREAD_COLOR)
img = cv.resize(img, (750, 750))

# Function to build a Gaussian pyramid
def build_gaussian_pyramid(image, levels):
    gaussian_pyramid = [image]
    for i in range(levels):
        image = cv.pyrDown(image)  # Downsample the image
        gaussian_pyramid.append(image)
    return gaussian_pyramid

# Function to build a Laplacian pyramid
def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    levels = len(gaussian_pyramid)
    for i in range(levels - 1):
        next_level_up = cv.pyrUp(gaussian_pyramid[i + 1])  # Upsample the next level
        next_level_up = cv.resize(next_level_up, (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        laplacian = cv.subtract(gaussian_pyramid[i], next_level_up)  # Subtract to get details
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])  # Add the smallest Gaussian level to Laplacian pyramid
    return laplacian_pyramid

# Function to display Gaussian and Laplacian pyramids
def show_pyramids(gaussian_pyramid, laplacian_pyramid):
    levels = len(gaussian_pyramid)
    fig, axs = plt.subplots(2, levels, figsize=(20, 8))

    for i in range(levels):
        axs[0, i].imshow(cv.cvtColor(gaussian_pyramid[i], cv.COLOR_BGR2RGB))
        axs[0, i].set_title(f'Gaussian Level {i}')

        axs[1, i].imshow(cv.cvtColor(np.abs(laplacian_pyramid[i]), cv.COLOR_BGR2RGB))
        axs[1, i].set_title(f'Laplacian Level {i}')

    plt.show()

# Number of levels in the pyramid
levels = 5

# Build Gaussian and Laplacian pyramids
gaussian_pyramid = build_gaussian_pyramid(img, levels)
laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)

# Display the pyramids
show_pyramids(gaussian_pyramid, laplacian_pyramid)
