import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
import matplotlib

# Paths
input_path = 'sample-images/'
output_path = 'image-processing/excercises/'

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



median_filtered = apply_median_blur(img_fg, 17)

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(cv.cvtColor(median_filtered, cv.COLOR_BGRA2GRAY), connectivity=8)

median_detected = median_filtered.copy()
img_detected = img.copy()

for i in range(1, num_labels):  # Skip background label 0
    centroid_x, centroid_y = centroids[i]
    x, y, w, h = stats[i][cv.CC_STAT_LEFT], stats[i][cv.CC_STAT_TOP], stats[i][cv.CC_STAT_WIDTH], stats[i][cv.CC_STAT_HEIGHT]
    aspect_ratio_tolerance = 0.2

    thickness = 10
    if abs(w / h - 1) < aspect_ratio_tolerance:
        cv.rectangle(median_detected, (int(x), int(y)), (int(x + w), int(y + h)), (218, 112, 214), thickness-5)
        cv.rectangle(img_detected, (int(x), int(y)), (int(x + w), int(y + h)), (218, 112, 214), thickness-5)
    cv.circle(median_detected, (int(centroid_x), int(centroid_y)), 5, (0, 255, 0), thickness)
    cv.circle(img_detected, (int(centroid_x), int(centroid_y)), 5, (0, 255, 0), thickness)



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
plt.imshow(cv.cvtColor(median_filtered, cv.COLOR_BGR2RGB))
plt.title('Bilateral Filtered Image')

plt.subplot(2, 3, 5)
plt.imshow(cv.cvtColor(median_detected, cv.COLOR_BGR2RGB))
plt.title('Median Detected Image')

plt.subplot(2, 3, 6)
plt.imshow(cv.cvtColor(img_detected, cv.COLOR_BGR2RGB))
plt.title('Object Detected Image')

plt.suptitle('Filtered Image with Centroids')
plt.tight_layout()

plt.savefig(os.path.join(output_path, 'output-detection.png'))
plt.show()


# cv.imshow('img', img_detected)
# cv.waitKey(0)
# cv.destroyAllWindows()
