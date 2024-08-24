from PIL import Image
import numpy as np
from scipy.signal import convolve2d


def lowpass_mean(input_file, output_file, filter_size):
    # Load the image in grayscale
    image = Image.open(input_file).convert("L")
    image_array = np.array(image)

    # Define the filter h[n1, n2]
    array = [[1] * filter_size] * filter_size
    h = np.array(array, dtype=np.float32)

    # Normalize the filter
    h = h / h.sum() # 1/9 * h[n1,n2]

    # Perform 2D convolution using scipy's convolve2d
    output_array = convolve2d(image_array, h, mode='same', boundary='fill', fillvalue=0)

    # Convert the output to an image
    output_image = Image.fromarray(np.uint8(output_array))

    # Save the output image
    output_image.save(output_file)


if __name__ == '__main__':
    input_file = "lena.jpg"
    output1 = "output_lena_1.jpg"
    output2 = "output_lena_2.jpg"
    output3 = "output_lena_3.jpg"
    lowpass_mean(input_file, output1, 3)
    lowpass_mean(input_file, output2, 5)
    lowpass_mean(input_file, output3, 7)

