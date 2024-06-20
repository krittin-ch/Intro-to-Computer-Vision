import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.signal
import os

input_path = 'sample-images/'
output_path = 'image-processing/pyramid/'

# Read and resize the image
img = cv.imread(os.path.join(input_path, 'image1.jpg'), cv.IMREAD_COLOR)
img = cv.resize(img, (750, 750))

# Downsampling function
def downsampling(large_img):
    small_to_large_image_size_ratio = 0.5
    small_img = cv.resize(large_img,
                          (0,0), 
                          fx=small_to_large_image_size_ratio, 
                          fy=small_to_large_image_size_ratio, 
                          interpolation=cv.INTER_NEAREST)
    return small_img

# Get image dimensions
rows, cols, _channels = img.shape

# Define the filters and apply them
bilinear_filter = np.ones((3,3), np.float32) / 9
gaussian_filter = scipy.signal.convolve2d(bilinear_filter, bilinear_filter, mode='full')
d_samp1 = cv.filter2D(img, -1, gaussian_filter)
d_samp1 = downsampling(d_samp1)

blur = cv.GaussianBlur(img, (5, 5), 0)
d_samp2 = downsampling(blur)

d_samp3 = cv.pyrDown(img, dstsize=(cols // 2, rows // 2))

# Create a 1x4 subplot to display the images
fig, ax = plt.subplots(1, 4, figsize=(20, 5))

ax[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
ax[0].set_title('Original Image')

ax[1].imshow(cv.cvtColor(d_samp1, cv.COLOR_BGR2RGB))
ax[1].set_title('Filtered & Downsampled (Gaussian)')

ax[2].imshow(cv.cvtColor(d_samp2, cv.COLOR_BGR2RGB))
ax[2].set_title('Blurred & Downsampled')

ax[3].imshow(cv.cvtColor(d_samp3, cv.COLOR_BGR2RGB))
ax[3].set_title('PyrDown')

plt.show()
