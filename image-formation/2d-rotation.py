from PIL import Image
import numpy as np
import cv2 as cv
import collections

img_path = 'sample-images/'

img = cv.imread(img_path + 'dog.jpg', cv.IMREAD_GRAYSCALE)
imgData = np.asarray(img)

rows, cols = img.shape # (1476, 1107)

theta = np.pi/2

# Scaling
a = 1
b = 0.2

rotation_matrix = np.array([
    [a*np.cos(theta), b*-np.sin(theta), cols],
    [a*np.sin(theta), b*np.cos(theta), 0]
    ])

M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0), -90, 1) # Built-in Rotation Matrix Generation

dst1 = cv.warpAffine(img, rotation_matrix, (cols, rows))

# Image Rotation with openCV
dst2 = cv.warpAffine(img, rotation_matrix, (cols,rows))

cv.imshow('img 1', dst1)
cv.imshow('img 2', dst2)
cv.waitKey(0)
cv.destroyAllWindows()

