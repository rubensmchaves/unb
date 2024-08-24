import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image in grayscale
image = Image.open('input.jpg').convert('L')
image_array = np.array(image)

# Step 1: Compute the DFT of the image
dft = np.fft.fft2(image_array)
dft_shifted = np.fft.fftshift(dft)  # Shift the zero frequency component to the center

# Get image dimensions
rows, cols = image_array.shape
crow, ccol = rows // 2 , cols // 2  # Center of the frequency domain

# Step 2: Create a low-pass filter in the frequency domain
# Cutoff frequency in terms of pixels, corresponding to omega = pi/2
cutoff = min(crow, ccol) // 2

# Create a mask with 1s inside the cutoff and 0s outside
mask = np.zeros((rows, cols), np.float32)
for i in range(rows):
    for j in range(cols):
        if (i - crow)**2 + (j - ccol)**2 <= cutoff**2:
            mask[i, j] = 1

# Step 3: Apply the mask to the shifted DFT
dft_shifted_filtered = dft_shifted * mask

# Step 4: Compute the inverse DFT
dft_filtered = np.fft.ifftshift(dft_shifted_filtered)  # Inverse shift
filtered_image_array = np.fft.ifft2(dft_filtered)
filtered_image_array = np.abs(filtered_image_array)

# Normalize the result to the 0-255 range for display
filtered_image_array = (filtered_image_array / np.max(filtered_image_array)) * 255
filtered_image_array = filtered_image_array.astype(np.uint8)

# Convert the result back to an image and save it
filtered_image = Image.fromarray(filtered_image_array)
filtered_image.save('output_filtered.jpg')

# Display the original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_array, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image_array, cmap='gray')
plt.title('Filtered Image (Low-pass)')

plt.show()
