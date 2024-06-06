import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# Input and output paths
input_path = 'sample-images/'
output_path = 'image-processing/excercises/'

# Read and resize the image (BGR format)
img_bgr = cv.imread(os.path.join(input_path, 'image1.jpg'))
img_bgr = cv.resize(img_bgr, (10, 10))

# Convert BGR to RGB
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

# Normalize the RGB image
img_rgb_normalized = img_rgb / 255.0

# Define the gamma value and its inverse
gamma = 2.2
inverse_gamma = 1.0 / gamma

# Apply inverse gamma correction (gamma expansion)
img_inverse_gamma_corrected = np.power(img_rgb_normalized, inverse_gamma)

# Transformation matrix from RGB to XYZ
rgb_to_xyz_matrix = np.array([[0.412453, 0.357580, 0.180423],
                              [0.212671, 0.715160, 0.072169],
                              [0.019334, 0.119193, 0.950227]])

# Apply the transformation to the linearized image
img_xyz = np.dot(img_inverse_gamma_corrected, rgb_to_xyz_matrix.T)

# Rescale XYZ image back to [0, 255] for visualization (optional, depends on use case)
img_xyz_display = np.clip(img_xyz * 255, 0, 255).astype(np.uint8)

# Display the original, gamma corrected, and XYZ images
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title('Original RGB Image')

plt.subplot(1, 3, 2)
plt.imshow((img_inverse_gamma_corrected * 255).astype(np.uint8))
plt.title('Inverse Gamma Corrected Image')

plt.subplot(1, 3, 3)
plt.imshow(img_xyz_display)
plt.title('Transformed XYZ Image')

plt.show()
