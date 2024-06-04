import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = 'C:/Users/kritt/Documents/GitHub/Intro-to-Computer-Vision/image-processing/'

path = os.path.join(base_dir, 'sample-images/')
output_path = os.path.join(base_dir, 'summed-area-table/')

img = cv.imread(os.path.join(path, 'image1.jpg'), cv.IMREAD_GRAYSCALE)
image = cv.resize(img, (100, 100))

integral_image = cv.integral(image)

# Function to calculate the sum of pixel values in a given rectangular region
def sum_region(integral_img, top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    total = (integral_img[y2 + 1, x2 + 1]
             - integral_img[y1, x2 + 1]
             - integral_img[y2 + 1, x1]
             + integral_img[y1, x1])
    return total

top_left = (0, 0)
bottom_right = (20, 20)

plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")

plt.subplot(1,2,2)
plt.imshow(integral_image.astype(np.uint8), cmap='gray')
plt.title("Integral Image")

image_size = f"Image Size: {image.shape[1]}x{image.shape[0]}"
plt.suptitle(f"Original and Integral Images\n{image_size}")

output_filename = f'output-{image.shape[1]}'

plt.savefig(os.path.join(output_path, output_filename + '.png'))

plt.show()
