import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.signal
import os

input_path = 'sample-images/'
output_path = 'image-processing/filters/'

img = cv.imread(os.path.join(input_path, 'image3.jpg'), cv.COLOR_BGR2RGB)
img = cv.resize(img, (750, 750))

sobel_filter_x = np.array([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]]) / 8

sobel_filter_y = sobel_filter_x.T

corner_filter = np.array([[-1, -2, 1], 
                          [-2, 4, -2], 
                          [-1, -2, 1]]) / 4

bilinear_filter = np.array([[1, 1, 1], 
                     [1, 1, 1], 
                     [1, 1, 1]]) / 16

edge_filter = np.array([[-1, -1, -1], 
                        [-1, 8, -1], 
                        [-1, -1, -1]]) 

gaussian_filter = scipy.signal.convolve2d(bilinear_filter, bilinear_filter, mode='full')
'''
gaussian_filter =
[[1. 2. 3. 2. 1.]
 [2. 4. 6. 4. 2.]
 [3. 6. 9. 6. 3.]
 [2. 4. 6. 4. 2.]
 [1. 2. 3. 2. 1.]] / 256
'''
def apply_filter_to_color_image(image, filter_kernel):
    channels = cv.split(image)
    filtered_channels = [scipy.signal.convolve2d(channel, filter_kernel, mode='same') for channel in channels]
    return cv.merge(filtered_channels)

def laplacian_of_gaussian(image, kernel_size=5, sigma=1.0):
    blurred = cv.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    laplacian = cv.Laplacian(blurred, cv.CV_64F)
    return laplacian

def second_order_laplacian_of_gaussian(image, kernel_size=5, sigma=1.0):
    blurred = cv.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    laplacian = cv.Laplacian(blurred, cv.CV_64F)
    second_order_laplacian = cv.Laplacian(laplacian, cv.CV_64F)
    return second_order_laplacian

def n_order_laplacian_of_gaussian(image, n=1, kernel_size=5, sigma=1.0):
    image = cv.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    for _ in range(n):
        image = cv.Laplacian(image, cv.CV_64F)
    return image

flt_img1 = apply_filter_to_color_image(img, sobel_filter_x)
flt_img2 = apply_filter_to_color_image(img, sobel_filter_y)
flt_img3 = flt_img1 + flt_img2

min_val, max_val = np.min(flt_img3), np.max(flt_img3)
normalized_img = (flt_img3 - min_val) / (max_val - min_val) * 255
flt_img3 = normalized_img.astype(np.uint8)

flt_img4 = apply_filter_to_color_image(img, corner_filter)
flt_img5 = apply_filter_to_color_image(img, gaussian_filter)

sharpening_factor = 0.1
flt_img6 = img + sharpening_factor * (img - flt_img5)

flt_img7 = laplacian_of_gaussian(img)
flt_img8 = second_order_laplacian_of_gaussian(img)
flt_img9 = n_order_laplacian_of_gaussian(img, n=10)

filters = [
    ("Original Image", img),
    ("Sobel Filter X", flt_img1),
    ("Sobel Filter Y", flt_img2),
    ("Combined Sobel Filters", flt_img3),
    ("Corner Filter", flt_img4),
    ("Gaussian Filter", flt_img5),
    ("Sharpened Image", flt_img6),
    ("Laplacian of Gaussian", flt_img7),
    ("Second Order Laplacian", flt_img8),
    ("10th Order Laplacian", flt_img9)
]

plt.figure(figsize=(15, 15))
plt.suptitle("Original Image and Filters", fontsize=14)

rows = 2
cols = 5

for i, (title, image) in enumerate(filters, start=1):
    plt.subplot(rows, cols, i)
    plt.imshow(cv.cvtColor(image.astype(np.uint8), cv.COLOR_BGR2RGB))
    plt.title(title, fontsize=10)
    plt.axis('on')

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'output-filters-1.png'))
plt.show()
