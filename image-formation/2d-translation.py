from PIL import Image
import numpy as np
import cv2 as cv
import collections

img_path = 'image-formation/sample_images/'

# img = Image.open(img_path + 'dog.jpg')
img = cv.imread(img_path + 'dog.jpg', cv.IMREAD_GRAYSCALE)
imgData = np.asarray(img)

iden = np.identity(2)
tx = 100
ty = 200
t = np.array([[tx, ty]]).T ## Vertical Axis

rows, cols = img.shape # (1476, 1107)

x, y = np.meshgrid(np.arange(cols), np.arange(rows))
coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x.ravel())])

translation_matrix = np.concatenate((iden, t), axis=1)

transformed_coords = translation_matrix @ coords
x_trans, y_trans = transformed_coords

def image_mapping(cols, rows, x_trans, y_trans, img):
    prev_x = (0, cols-1)
    prev_y = (0, rows-1)
    new_x = (min(x_trans), max(x_trans))
    new_y = (min(y_trans), max(y_trans))
    
    x0 = int(max(prev_x[0], new_x[0])) # prev_x[0] <= new_x[0]
    y0 = int(max(prev_y[0], new_y[0])) # prev_y[0] <= new_y[0]
    
    xn = int(min(prev_x[1], new_x[1])) # new_x[1] <= prev_x[1]
    yn = int(min(prev_y[1], new_y[1])) # new_y[1] <= prev_y[1]

    img_width = img.shape[1]
    img_height = img.shape[0]

    transformed_img = np.copy(img)

    transformed_img[:, :x0] = np.zeros((img_height, x0)) 
    transformed_img[:y0, :] = np.zeros((y0, img_width)) 
    transformed_img[:, xn:] = np.zeros((img_height, img_width - xn)) 
    transformed_img[yn:, :] = np.zeros((img_height - yn, img_width))

    return transformed_img 
    
translated_img = image_mapping(cols, rows, x_trans, y_trans, img)
# cv.imshow('img 1', translated_img)
# cv.imshow('img 2', img)

# Image Translation with openCV
dst = cv.warpAffine(img, translation_matrix, (cols,rows))

# cv.imshow('img 3', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

