import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths
input_path = 'sample-images/'
output_path = 'image-processing/excercises/'

# Read and resize the image
img = cv.imread(os.path.join(input_path, 'view.jpg'))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

l = 1000
img = cv.resize(img, (l, l))

fig, ax = plt.subplots(2, 2, figsize=(15, 15))

# Display the original image
ax[0, 0].imshow(img)
ax[0, 0].set_title("Original Image")

# Plot the color histogram for the original image
B_histo = cv.calcHist([img],[0], None, [256], [0,256])
G_histo = cv.calcHist([img],[1], None, [256], [0,256])
R_histo = cv.calcHist([img],[2], None, [256], [0,256])

for hist, color in zip([B_histo, G_histo, R_histo], ['b', 'g', 'r']):
    ax[1, 0].plot(hist, color=color)
ax[1, 0].set_title("Color Histogram (Original)")
ax[1, 0].set_xlabel("Color value")
ax[1, 0].set_ylabel("Pixel count")

# Equalize each color channel separately
B = img[:,:,0] 
G = img[:,:,1] 
R = img[:,:,2] 

b_equi = cv.equalizeHist(B)
g_equi = cv.equalizeHist(G)
r_equi = cv.equalizeHist(R)

# Merge and display the equalized image
equi_im = cv.merge([b_equi, g_equi, r_equi])

ax[0, 1].imshow(equi_im)
ax[0, 1].set_title("Equalized Image")
ax[0, 1].set_xlabel("Width")
ax[0, 1].set_ylabel("Height")

# Plot the color histograms after equalization
B_histo_eq = cv.calcHist([equi_im],[0], None, [256], [0,256])
G_histo_eq = cv.calcHist([equi_im],[1], None, [256], [0,256])
R_histo_eq = cv.calcHist([equi_im],[2], None, [256], [0,256])

for hist, color in zip([B_histo_eq, G_histo_eq, R_histo_eq], ['b', 'g', 'r']):
    ax[1, 1].plot(hist, color=color)
ax[1, 1].set_title("Color Histogram (Equalized)")
ax[1, 1].set_xlabel("Color value")
ax[1, 1].set_ylabel("Pixel count")


plt.savefig(os.path.join(output_path, 'output-histogram-equalization-2.jpg'))

plt.tight_layout()
plt.show()
