import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy

# Paths
input_path = 'sample-images/'
output_path = 'image-processing/filters/'

# Read and resize the image
img_bg = cv.imread(os.path.join(input_path, 'view-bg.jpg'), cv.IMREAD_COLOR)
img = cv.imread(os.path.join(input_path, 'view.jpg'), cv.IMREAD_COLOR)

l = 1024

img = cv.resize(img, (l, l))
img_bg = cv.resize(img_bg, (l, l))

# Define the corner filter
corner_filter = np.array([[-1, -2, 1], 
                          [-2, 4, -2], 
                          [-1, -2, 1]]) / 4

def apply_filter_to_color_image(image, filter_kernel):
    channels = cv.split(image)
    filtered_channels = [scipy.signal.convolve2d(channel, filter_kernel, mode='same') for channel in channels]
    return cv.merge(filtered_channels)

def apply_variance_check(img, img_bg, v):
    new_img = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            var = ((img[i,j] - img_bg[i,j])^2)/255
            if np.all(var < v):
                new_img[i,j] = img[i,j]
            else:
                new_img[i,j] = 0
    return new_img.astype(np.uint8)

img_fg = apply_variance_check(img, img_bg, 0.925)

def apply_median_blur(img_fg, kernel):
    b, g, r = cv.split(img_fg)
    b = cv.medianBlur(b, kernel)
    g = cv.medianBlur(g, kernel)
    r = cv.medianBlur(r, kernel)
    return cv.merge([b, g, r])


bilateral_filtered = cv.bilateralFilter(img_fg, 9, 75, 75)

median_filtered = apply_median_blur(img_fg, 17)
median_filtered = apply_median_blur(median_filtered, 5)

nlm_denoised = cv.fastNlMeansDenoisingColored(img_fg, None, 10, 10, 7, 21)


num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(cv.cvtColor(median_filtered, cv.COLOR_BGRA2GRAY), connectivity=8)

print(num_labels, labels, stats, centroids)

for i in range(1, num_labels):  # Skip background label 0
    centroid_x, centroid_y = centroids[i]
    print(f"Centroid of object {i}: ({centroid_x}, {centroid_y})")

# Visualize the centroids on the image
plt.imshow(cv.cvtColor(median_filtered, cv.COLOR_GRAY2RGB))
for i in range(1, num_labels):  # Skip background label 0
    centroid_x, centroid_y = centroids[i]
    plt.scatter(centroid_x, centroid_y, color='red', s=30)
plt.title('Filtered Image with Centroids')
plt.show()
'''
# Set figure size
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(2, 3, 2)
plt.imshow(cv.cvtColor(img_bg, cv.COLOR_BGR2RGB))
plt.title('Background Image')

plt.subplot(2, 3, 3)
plt.imshow(cv.cvtColor(img_fg, cv.COLOR_BGR2RGB))
plt.title('Background Removed Image')

plt.subplot(2, 3, 4)
plt.imshow(cv.cvtColor(bilateral_filtered, cv.COLOR_BGR2RGB))
plt.title('Bilateral Filtered Image')

plt.subplot(2, 3, 5)
plt.imshow(cv.cvtColor(median_filtered, cv.COLOR_BGR2RGB))
plt.title('Median Filtered Image')

plt.subplot(2, 3, 6)
plt.imshow(cv.cvtColor(nlm_denoised, cv.COLOR_BGR2RGB))
plt.title('NLM Denoised Image')

plt.tight_layout()
plt.show()

'''