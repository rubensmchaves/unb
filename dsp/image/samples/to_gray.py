from PIL import Image
import numpy as np
from scipy.signal import convolve2d

# Load the image in grayscale
image = Image.open('lena.png').convert('L')
image_array = np.array(image)

# Convert the output to an image
output_image = Image.fromarray(np.uint8(image_array))

# Save the output image
output_image.save('lena_mono.png')
