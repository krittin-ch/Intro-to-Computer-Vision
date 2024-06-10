import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
# Paths
input_path = 'sample-images/'
output_path = 'image-processing/filters/'

# Read and resize the image
img_bg = cv.imread(os.path.join(input_path, 'view-bg.jpg'), cv.IMREAD_COLOR)
img = cv.imread(os.path.join(input_path, 'view.jpg'), cv.IMREAD_COLOR)

l = 1024

img = cv.resize(img, (l, l))
img_bg = cv.resize(img_bg, (l, l))
# img_next = cv.resize(img_next, (l, l))

def compute_background(image_sequence):
    """
    Compute the mean and variance of each pixel in the input image sequence.
    
    Parameters:
        image_sequence (list of numpy arrays): Input image sequence.
        
    Returns:
        background_mean (numpy array): Mean image representing the background.
        background_variance (numpy array): Variance image representing the background.
    """
    # Convert image sequence to numpy array
    image_sequence = np.array(image_sequence)
    
    # Compute mean and variance along the time axis
    background_mean = np.mean(image_sequence, axis=0)
    background_variance = np.var(image_sequence, axis=0)
    
    return background_mean, background_variance

def classify_foreground(background_mean, background_variance, new_frame, threshold=50):
    """
    Classify each pixel in the new frame as foreground or background based on the mean and variance.
    
    Parameters:
        background_mean (numpy array): Mean image representing the background.
        background_variance (numpy array): Variance image representing the background.
        new_frame (numpy array): New frame to classify.
        threshold (int): Threshold value for classifying foreground pixels.
        
    Returns:
        foreground_mask (numpy array): Binary mask indicating foreground pixels (1) and background pixels (0).
    """
    # Compute absolute difference between new frame and background mean
    abs_diff = np.abs(new_frame - background_mean)
    
    # abs_diff = np.abs(new_frame - background_variance)
    
    # Create binary mask based on threshold
    foreground_mask = np.where(abs_diff > threshold, 1, 0)
    
    return foreground_mask

background_mean, background_variance = compute_background(img_bg)

def adjust_foreground_mask(foreground_mask):
    # Create a mask for pixels where any R, G, or B value is < 100
    # low_mask = np.any(foreground_mask < 200, axis=-1)
    # Create a mask for pixels where any R, G, or B value is > 200
    high_mask = np.any(foreground_mask > 200, axis=-1)
    
    # Set these pixels to 0
    # foreground_mask[low_mask] = 0
    # Set these pixels to 255
    foreground_mask[high_mask] = 255
    
    return foreground_mask


corner_filter = np.array([[-1, -2, 1], 
                          [-2, 4, -2], 
                          [-1, -2, 1]]) / 4

def apply_filter_to_color_image(image, filter_kernel):
    channels = cv.split(image)
    filtered_channels = [scipy.signal.convolve2d(channel, filter_kernel, mode='same') for channel in channels]
    return cv.merge(filtered_channels)

foreground_mask = abs(img - img_bg)
foreground_mask = np.uint8(foreground_mask)

# foreground_mask = adjust_foreground_mask(foreground_mask)

foreground_mask = apply_filter_to_color_image(adjust_foreground_mask(foreground_mask), corner_filter).astype(np.uint8)

# print(foreground_mask)

plt.subplot(1,3,1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1,3,2)
plt.imshow(cv.cvtColor(img_bg, cv.COLOR_BGR2RGB))
plt.title('Backround Image')

plt.subplot(1,3,3)
plt.imshow(cv.cvtColor(foreground_mask, cv.COLOR_BGR2RGB))
plt.title('Background Removed Image')

plt.show()

# # Example usage
# if __name__ == "__main__":
#     # Load input video sequence
#     video_sequence = [...]  # Load or capture video frames
    
#     # Compute background model
#     background_mean, background_variance = compute_background(video_sequence)
    
#     # Process each frame in the video sequence
#     for frame in video_sequence:
#         # Classify foreground pixels
#         foreground_mask = classify_foreground(background_mean, background_variance, frame)
        
#         # Optionally, perform further processing such as alpha channel computation, compositing, morphology, etc.
