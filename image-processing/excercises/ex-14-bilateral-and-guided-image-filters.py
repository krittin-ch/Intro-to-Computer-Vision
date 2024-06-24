import cv2 as cv
import numpy as np

def guided_filter(I, p, radius, eps):
    """Perform guided filtering.
    
    Args:
        I: guidance image (grayscale or RGB)
        p: filtering input image (grayscale or RGB)
        radius: the radius of the window (integer)
        eps: regularization parameter (float)
        
    Returns:
        q: the filtered image
    """
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    
    # Calculate the means
    mean_I = cv.boxFilter(I, cv.CV_32F, (radius, radius))
    mean_p = cv.boxFilter(p, cv.CV_32F, (radius, radius))
    mean_Ip = cv.boxFilter(I * p, cv.CV_32F, (radius, radius))
    
    # Calculate the covariance of (I, p) in each local patch
    cov_Ip = mean_Ip - mean_I * mean_p
    
    # Calculate the variance of I in each local patch
    mean_II = cv.boxFilter(I * I, cv.CV_32F, (radius, radius))
    var_I = mean_II - mean_I * mean_I
    
    # Calculate the linear coefficients a and b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    # Calculate the means of a and b
    mean_a = cv.boxFilter(a, cv.CV_32F, (radius, radius))
    mean_b = cv.boxFilter(b, cv.CV_32F, (radius, radius))
    
    # Calculate the output image q
    q = mean_a * I + mean_b
    
    return q


import matplotlib.pyplot as plt
import os

# Paths
input_path = 'sample-images/'
output_path = 'image-processing/filters/'

# Read and resize the image
I = cv.imread(os.path.join(input_path, 'view.jpg'), cv.IMREAD_COLOR)
p = cv.imread(os.path.join(input_path, 'dog.jpg'), cv.IMREAD_COLOR)

I = cv.resize(I, (1000, 1000))
p = cv.resize(p, (1000, 1000))

# Convert images to grayscale if they are in color
if len(I.shape) == 3:
    I_gray = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
else:
    I_gray = I

if len(p.shape) == 3:
    p_gray = cv.cvtColor(p, cv.COLOR_BGR2GRAY)
else:
    p_gray = p

# Set parameters for the guided filter
radius = 8  # Radius of the window
eps = 0.1**2  # Regularization parameter

# Apply the guided filter
q = guided_filter(I_gray, p_gray, radius, eps)

# Display the original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Guidance Image")
plt.imshow(I_gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Input Image")
plt.imshow(p_gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Filtered Image")
plt.imshow(q, cmap='gray')
plt.axis('off')

plt.show()
