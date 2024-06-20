import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

input_path = 'sample-images/'
output_path = 'image-processing/filters/'

# Read and resize the image
img = cv.imread(os.path.join(input_path, 'image1.jpg'), cv.IMREAD_COLOR)
img = cv.resize(img, (1000, 1000))

img = np.array(img)

def downsampling(large_img):
    """
    Perform downsampling of the input image using nearest-neighbor interpolation.
    
    Parameters:
    large_img (numpy.ndarray): The input image to be downsampled.
    
    Returns:
    numpy.ndarray: The downsampled image.
    """
    small_to_large_image_size_ratio = 0.5
    small_img = cv.resize(large_img,
                          (0,0), 
                          fx=small_to_large_image_size_ratio, 
                          fy=small_to_large_image_size_ratio, 
                          interpolation=cv.INTER_NEAREST)
    return small_img

def upsampling(small_img):
    """
    Perform upsampling of the input image using linear interpolation.
    
    Parameters:
    small_img (numpy.ndarray): The input image to be upsampled.
    
    Returns:
    numpy.ndarray: The upsampled image.
    """
    large_to_small_image_size_ratio = 2
    large_img = cv.resize(small_img,
                          (0, 0), 
                          fx=large_to_small_image_size_ratio, 
                          fy=large_to_small_image_size_ratio, 
                          interpolation=cv.INTER_LINEAR)
    return large_img

def gaussian_pyramid(image):
    """
    Construct a Gaussian pyramid by applying Gaussian blur and downsampling.
    
    Parameters:
    image (numpy.ndarray): The input image.
    
    Returns:
    numpy.ndarray: The processed image after Gaussian blur and downsampling.
    """
    image = cv.GaussianBlur(image, (3,3), 0)
    image = downsampling(image)
    return image

def band_pass(image, blurred_image):
    """
    Compute the band-pass image by upsampling the blurred image and subtracting from the original image.
    
    Parameters:
    image (numpy.ndarray): The original input image.
    blurred_image (numpy.ndarray): The blurred image.
    
    Returns:
    numpy.ndarray: The band-pass image.
    """
    blurred_image = upsampling(blurred_image)
    image = cv.subtract(blurred_image, image)
    return image

def pyramid_block(image):
    """
    Generate a single level of the Laplacian pyramid.
    
    Parameters:
    image (numpy.ndarray): The input image.
    
    Returns:
    tuple: A tuple containing the band-passed image and the blurred image.
    """
    blurred_image = gaussian_pyramid(image)
    bandpassed_image = band_pass(image, blurred_image)
    return bandpassed_image, blurred_image

def quantization_block(image):
    """
    Apply quantization (median and bilateral filtering) to the input image.
    
    Parameters:
    image (numpy.ndarray): The input image.
    
    Returns:
    numpy.ndarray: The quantized image after median and bilateral filtering.
    """
    median_filtered = cv.medianBlur(image, 5)
    bilateral_filtered = cv.bilateralFilter(image, 9, 75, 75)
    return bilateral_filtered

def reconstruction_block(blurred_image, bandpassed_image):
    """
    Reconstruct the image from the Laplacian pyramid level.
    
    Parameters:
    blurred_image (numpy.ndarray): The blurred image.
    bandpassed_image (numpy.ndarray): The band-passed image.
    
    Returns:
    numpy.ndarray: The reconstructed image.
    """
    blurred_image = upsampling(blurred_image)
    return cv.add(bandpassed_image, blurred_image)

def laplacian_pyramid(image):
    """
    Constructs a Laplacian pyramid for the input image.

    Parameters:
    image (numpy.ndarray): The input image to construct the Laplacian pyramid.

    Returns:
    numpy.ndarray: The final reconstructed image from the Laplacian pyramid.
    """
    bandpassed_image1, blurred_image1 = pyramid_block(image)
    bandpassed_image2, blurred_image2 = pyramid_block(blurred_image1)

    quantized_bandpassed_image1 = quantization_block(bandpassed_image1)   
    quantized_bandpassed_image2 = quantization_block(bandpassed_image2)    
    quantized_blurred_image2 = quantization_block(blurred_image2)

    reconstructed_image1 = reconstruction_block(quantized_blurred_image2, quantized_bandpassed_image2)
    final_output = reconstruction_block(reconstructed_image1, quantized_bandpassed_image1)

    return final_output


def laplacian_pyramid_levels(image, levels):
    """
    Build a Laplacian pyramid with the specified number of levels.
    
    Parameters:
    image (numpy.ndarray): The input image.
    levels (int): Number of pyramid levels.
    
    Returns:
    numpy.ndarray: The reconstructed image from the Laplacian pyramid.
    """
    pyramid_levels = []
    current_image = image.copy()

    for level in range(levels):
        bandpassed_image, blurred_image = pyramid_block(current_image)
        quantized_bandpassed_image = quantization_block(bandpassed_image)
        quantized_blurred_image = quantization_block(blurred_image)
        
        pyramid_levels.append((quantized_bandpassed_image, quantized_blurred_image))
        
        current_image = blurred_image

    reconstructed_image = pyramid_levels[-1][1]  
    
    for level in range(levels - 1, 0, -1):
        quantized_blurred_image, quantized_bandpassed_image = pyramid_levels[level - 1]
        reconstructed_image = reconstruction_block(reconstructed_image, quantized_bandpassed_image)

    return reconstructed_image

level = 2
output_img2 = laplacian_pyramid_levels(img, level)
output_img1 = laplacian_pyramid(img)

# Plotting original image and two Laplacian pyramid reconstructions
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axs[0].set_title('Original Image')

axs[1].imshow(cv.cvtColor(output_img1, cv.COLOR_BGR2RGB))
axs[1].set_title('Laplacian Pyramid')

axs[2].imshow(cv.cvtColor(output_img2, cv.COLOR_BGR2RGB))
axs[2].set_title(f'Laplacian Pyramid ({level} levels)')

plt.tight_layout()

# Save the figure
output_file = os.path.join(output_path, 'laplacian_pyramid_reconstructions.png')
plt.savefig(output_file)

plt.show()
