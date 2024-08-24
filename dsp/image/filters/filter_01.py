from PIL import Image
import numpy as np
from scipy.signal import convolve2d

# Load the image in grayscale
image = Image.open('input.jpg')
image_array = np.array(image)

# Define the filter h[n1, n2]
h = np.array([[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]], dtype=np.float32)

# Normalize the filter
h = h / h.sum() # 1/9 * h[n1,n2]

# Perform 2D convolution using scipy's convolve2d
output_array = convolve2d(image_array, h, mode='same', boundary='fill', fillvalue=0)

# Convert the output to an image
output_image = Image.fromarray(np.uint8(output_array))

# Save the output image
output_image.save('output_filter_01.jpg')
