import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths
input_path = 'sample-images/'
output_path = 'image-processing/filters/'

# Read and resize the image
img_bg = cv.imread(os.path.join(input_path, 'green-bg-1.jpg'), cv.IMREAD_COLOR)
img = cv.imread(os.path.join(input_path, 'green-bg-2.jpg'), cv.IMREAD_COLOR)

img_bg = cv.resize(img_bg, (1000, 1000))
img = cv.resize(img, (1000, 1000))

ycrcb_img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

# Define the range for the green color in YCrCb space
lower_green = np.array([0, 133, 77])
upper_green = np.array([255, 173, 127])

# Create a mask to detect green color
mask = cv.inRange(ycrcb_img, lower_green, upper_green)

# Refine the mask using morphological operations
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

# Invert the mask to get the foreground
mask_inv = cv.bitwise_not(mask)

# Use the mask to extract the foreground and background
img_fg = cv.bitwise_and(img, img, mask=mask_inv)
img_bg = cv.bitwise_and(img_bg, img_bg, mask=mask)

# Combine the foreground and background
img_remv_bg = cv.add(img_fg, img_bg)

print(img_remv_bg)
plt.subplot(1,3,1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1,3,2)
plt.imshow(cv.cvtColor(img_bg, cv.COLOR_BGR2RGB))
plt.title('Backround Image')

plt.subplot(1,3,3)
plt.imshow(cv.cvtColor(img_remv_bg, cv.COLOR_BGR2RGB))
plt.title('Background Removed Image')

plt.show()