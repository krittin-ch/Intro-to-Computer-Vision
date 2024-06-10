import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths
input_path = 'sample-images/'
output_path = 'image-processing/filters/'

# Read and resize the image
img_bg = cv.imread(os.path.join(input_path, 'green-bg-1.jpg'), cv.IMREAD_COLOR)
img = cv.imread(os.path.join(input_path, 'green-bg-2.jpg'), cv.IMREAD_COLOR)
img_next = cv.imread(os.path.join(input_path, 'green-bg-3.jpg'), cv.IMREAD_COLOR)

l = 1024

img = cv.resize(img, (l, l))
img_bg = cv.resize(img_bg, (l, l))
img_next = cv.resize(img_next, (l, l))

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


# background_mean, background_variance = compute_background(img_bg)
# foreground_mask = classify_foreground(background_mean, background_variance, img_next, threshold=50)

foreground_mask = img_next - img_bg
foreground_mask = np.uint8(foreground_mask)

import torch

import segmentation_models_pytorch as smp

in_channels = 3 
out_channels = 1

model = smp.Unet(encoder_name="resnet34", in_channels=in_channels, classes=out_channels, encoder_weights="imagenet")

                 
image_tensor = torch.tensor(img_next, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

# Perform inference to get the segmentation mask
with torch.no_grad():
    output = model(image_tensor)

# Post-process the output to get the segmentation mask
# You may need to apply thresholding or other operations to get a binary mask
segmentation_mask = output.squeeze().cpu().numpy()
segmentation_mask = np.where(segmentation_mask > 0.5, 255, 0)  # Example thresholding, adjust as needed



plt.subplot(1,3,1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1,3,2)
plt.imshow(cv.cvtColor(img_bg, cv.COLOR_BGR2RGB))
plt.title('Backround Image')

plt.subplot(1,3,3)
plt.imshow(segmentation_mask, cmap='gray')
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
