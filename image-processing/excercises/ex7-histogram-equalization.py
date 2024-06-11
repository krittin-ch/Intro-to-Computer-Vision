import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths
input_path = 'sample-images/'
output_path = 'image-processing/filters/'

# Read and resize the image
img = cv.imread(os.path.join(input_path, 'view.jpg'), cv.COLOR_BGR2RGB)

l = 1000
img = cv.resize(img, (l, l))

red = np.zeros((l, l, 1))
green = np.zeros((l, l, 1))
blue = np.zeros((l, l, 1))

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        red[i, j] = img[i, j, 0] + (red[i, j-1] if j-1 >= 0 else 0)
        green[i, j] = img[i, j, 1] + (green[i, j-1] if j-1 >= 0 else 0)
        blue[i, j] = img[i, j, 2] + (blue[i, j-1] if j-1 >= 0 else 0)

# new_img = np.concatenate((red, green, blue), axis=2)

# red_flat = np.ravel(red)
# green_flat = np.ravel(green)
# blue_flat = np.ravel(blue)  

# # Create the line plot
# plt.figure(figsize=(10, 6))
# plt.plot(range(l*l), red_flat, 'r-', label='Red')
# plt.plot(range(l*l), green_flat, 'g-', label='Green')
# plt.plot(range(l*l), blue_flat, 'b-', label='Blue')
# plt.xlabel('Pixel Index')
# plt.ylabel('Accumulated Value')
# plt.title('Accumulated Color Values')
# plt.legend()
# plt.show()



# display the image
fig, ax = plt.subplots()


# tuple to select colors of each channel line
colors = ("red", "green", "blue")

# create the histogram plot, with three lines, one for
# each color
fig, ax = plt.subplots()
ax.set_xlim([0, 256])
for channel_id, color in enumerate(colors):
    histogram, bin_edges = np.histogram(
        img[:, :, channel_id], bins=256, range=(0, 256)
    )
    ax.plot(bin_edges[0:-1], histogram, color=color)

ax.set_title("Color Histogram")
ax.set_xlabel("Color value")
ax.set_ylabel("Pixel count")

ax.imshow(img)
