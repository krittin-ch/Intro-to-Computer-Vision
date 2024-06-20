import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv

import pywt
import pywt.data


# Paths
input_path = 'sample-images/'
output_path = 'image-processing/filters/'

# Load image
original = cv.imread(os.path.join(input_path, 'image1.jpg'), cv.IMREAD_GRAYSCALE)

l = 1024
original = cv.resize(original, (l, l))


# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()
