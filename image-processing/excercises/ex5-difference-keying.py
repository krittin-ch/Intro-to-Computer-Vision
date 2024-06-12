import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
import statistics

# Paths
input_path = 'sample-images/'
output_path = 'image-processing/filters/'

# Read and resize the image
img_bg = cv.imread(os.path.join(input_path, 'view-bg.jpg'), cv.IMREAD_COLOR)
img = cv.imread(os.path.join(input_path, 'view.jpg'), cv.IMREAD_COLOR)

l = 1024

img = cv.resize(img, (l, l))
img_bg = cv.resize(img_bg, (l, l))
# img_next = cv.resize(img_next, (l, l))


def adjust_foreground_mask(foreground_mask):
    high_mask = np.any(foreground_mask > 200, axis=-1)
    
    foreground_mask[high_mask] = 255
    
    return foreground_mask


corner_filter = np.array([[-1, -2, 1], 
                          [-2, 4, -2], 
                          [-1, -2, 1]]) / 4

def apply_filter_to_color_image(image, filter_kernel):
    channels = cv.split(image)
    filtered_channels = [scipy.signal.convolve2d(channel, filter_kernel, mode='same') for channel in channels]
    return cv.merge(filtered_channels)

# img = img[:300, :300]
# img_bg = img_bg[:300, :300]

def apply_variance_check(img, img_bg, v):
    new_img = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            var = ((img[i,j] - img_bg[i,j])^2)/255
            # print(var)
            if np.all(var < v):
                new_img[i,j] = img[i,j]
            else :
                new_img[i,j] = 0
    return new_img.astype(np.uint8)


img_fg = apply_variance_check(img, img_bg, 0.8)
bilateral_filtered = cv.bilateralFilter(img_fg, 9, 75, 75)


b, g, r = cv.split(img_fg)
b = cv.medianBlur(b, 5)
g = cv.medianBlur(g, 5)
r = cv.medianBlur(r, 5)
median_filtered = cv.merge([b, g, r])

nlm_denoised = cv.fastNlMeansDenoisingColored(img_fg, None, 10, 10, 7, 21)

plt.subplot(1,3,1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1,3,2)
plt.imshow(cv.cvtColor(img_bg, cv.COLOR_BGR2RGB))
plt.title('Backround Image')

plt.subplot(1,3,3)
plt.imshow(cv.cvtColor(img_fg, cv.COLOR_BGR2RGB))
plt.title('Background Removed Image')

plt.subplot(2,3,4)
plt.imshow(cv.cvtColor(bilateral_filtered, cv.COLOR_BGR2RGB))
plt.title('Background Removed Image')

plt.subplot(2,3,5)
plt.imshow(cv.cvtColor(median_filtered, cv.COLOR_BGR2RGB))
plt.title('Background Removed Image')

plt.subplot(2,3,6)
plt.imshow(cv.cvtColor(nlm_denoised, cv.COLOR_BGR2RGB))
plt.title('Background Removed Image')


plt.show()
