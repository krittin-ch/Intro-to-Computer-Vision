import cv2 as cv
import numpy as np

path = 'c:/Users/kritt/Documents/GitHub/Intro-to-Computer-Vision/image-formation/excercises/sample_images/'
img = cv.imread(path + 'rectangular.jpg', cv.IMREAD_COLOR)

rows, cols, ch = img.shape

pts1 = np.float32([[1, 0],
                   [0, 1], 
                   [0.03, 0.06]])


pts2 = np.float32([[1, 0],
                   [0, 1], 
                   [0.09, 0.12]])
 
M = cv.getAffineTransform(pts1, pts2)
dst = cv.warpAffine(img, M, (cols, rows))
 
cv.imshow('img 1', dst)
cv.imwrite('rectangular_affine.jpg', dst)
cv.waitKey(0)
cv.destroyAllWindows()
