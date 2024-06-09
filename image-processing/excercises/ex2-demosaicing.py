import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# Input and output paths
input_path = 'sample-images/'
output_path = 'image-processing/filters/'

# Read the RGB image
img_rgb = cv.imread(os.path.join(input_path, 'image1.jpg'))
img_rgb = cv.resize(img_rgb, (700, 700))

# Define the Bayer pattern arrangement (e.g., RGGB)
def apply_bayer_mosaic(img):
    bayer_mosaic = np.zeros(img.shape, dtype=np.uint8)
    bayer_mosaic[::2, ::2, 2] = img[::2, ::2, 2]  # B
    bayer_mosaic[::2, 1::2, 1] = img[::2, 1::2, 1]  # G1
    bayer_mosaic[1::2, ::2, 1] = img[1::2, ::2, 1]  # G2
    bayer_mosaic[1::2, 1::2, 0] = img[1::2, 1::2, 0]  # R
    return bayer_mosaic

def apply_cygm_mosaic(img):
    """
    Apply a CYGM mosaic pattern to an RGB image.

    Parameters:
        img (numpy.ndarray): Input RGB image.

    Returns:
        numpy.ndarray: CYGM mosaic image.
    """
    cygm_mosaic = np.zeros(img.shape, dtype=np.uint8)
    cygm_mosaic[::2, ::2, 1] = img[::2, ::2, 1]  # Green channel for Cyan
    cygm_mosaic[::2, ::2, 2] = img[::2, ::2, 2]  # Blue channel for Cyan
    
    cygm_mosaic[::2, 1::2, 0] = img[::2, 1::2, 0]  # Red channel for Yellow
    cygm_mosaic[::2, 1::2, 1] = img[::2, 1::2, 1]  # Green channel for Yellow
    
    cygm_mosaic[1::2, ::2, 1] = img[1::2, ::2, 1]  # Green channel for Green

    cygm_mosaic[1::2, 1::2, 0] = img[1::2, 1::2, 0]  # Red channel for Magenta
    cygm_mosaic[1::2, 1::2, 2] = img[1::2, 1::2, 2]  # Blue channel for Magenta
    return cygm_mosaic

img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

def apply_color_illusion(img):
    """
    Apply a color dithering pattern to a grayscale image to create the illusion of color.

    Parameters:
        img_gray (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Color dithered image.
    """
    # Create an empty image with 3 channels
    h, w, _ = img.shape
    color_dithered = np.zeros((h, w, 3), dtype=np.uint8)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Define a simple color pattern (CYM - Cyan, Yellow, Magenta)
    for y in range(h):
        for x in range(w):
            if (y%20 == 0):
                if np.all(img[y, x] == [255, 255, 255]):
                    color_dithered[y, x] = [150, 150, 150]
                else :
                    for i in range(-1,2):
                        # max_idx = np.argmax(img[y+i,x])
                        # color_dithered[y+i, x] = \
                        # [img[y+i,x, max_idx], 0, 0] if max_idx == 0 else \
                        # [0, img[y+i,x, max_idx], 0] if max_idx == 1 else \
                        # [0, 0, img[y+i,x, max_idx]]
                        color_dithered[y+i, x] = img[y+i, x]
            
            elif (x%20 == 0):
                if np.all(img[y, x] == [255, 255, 255]):
                    color_dithered[y, x] = [150, 150, 150]
                else :
                    for i in range(-1,2):
                        # max_idx = np.argmax(img[y,x+i])
                        # color_dithered[y, x+i] = \
                        # [img[y,x+i, max_idx], 0, 0] if max_idx == 0 else \
                        # [0, img[y,x+i, max_idx], 0] if max_idx == 1 else \
                        # [0, 0, img[y,x+i, max_idx]]
                        color_dithered[y+i, x] = img[y+i, x]
            else:
                color_dithered[y, x] = img_gray[y, x]

    return color_dithered
            
# Apply the color dithering pattern
color_dithered_image = apply_color_illusion(img_rgb)


bayer_mosaic = apply_bayer_mosaic(img_rgb)
cygm_mosaic = apply_cygm_mosaic(img_rgb)

cv.imshow('Image', color_dithered_image)

cv.waitKey(0)
cv.destroyAllWindows()


plt.subplot(2, 3, 1)
plt.imshow(cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB))
plt.title('Original RGB Image')

plt.subplot(2, 3, 2)
plt.imshow(cv.cvtColor(bayer_mosaic, cv.COLOR_BGR2RGB))
plt.title('Bayer Mosaic Pattern')

plt.subplot(2, 3, 3)
plt.imshow(cv.cvtColor(bayer_mosaic, cv.COLOR_BGR2GRAY), cmap='gray')
plt.title('Grayscale Bayer Mosaic')

plt.subplot(2, 3, 4)
plt.imshow(cygm_mosaic)
plt.title('CYGM Mosaic Pattern')

plt.subplot(2, 3, 5)
plt.imshow(cv.cvtColor(color_dithered_image, cv.COLOR_BGR2RGB))
plt.title('Color Dithered Image')

color_dithered_image
plt.show()
