from PIL import Image
import numpy as np
import math
from scipy.signal import convolve2d


def highpass_mean(input_file, output_file, filter_size):
    # Load the image in grayscale
    image = Image.open(input_file).convert("L")
    image_array = np.array(image)

    bit_depth = 8

    # Define the filter h[n1, n2]
    array = [[1] * filter_size] * filter_size
    h = np.array(array, dtype=np.float32)

    # Normalize the filter and convert to highpass filter
    h = -(h / h.sum())
    i = filter_size // 2
    h[i][i] = h[i][i] + 1
    
    # Perform 2D convolution using scipy's convolve2d
    output_array = convolve2d(image_array, h, mode='same', boundary='fill', fillvalue=0)

    adjust = math.pow(2, bit_depth)/2 
    output_array = (output_array / 2) + adjust
    print('Adjust:', adjust)

    # Convert the output to an image
    output_image = Image.fromarray(np.uint8(output_array))

    # Save the output image
    output_image.save(output_file)


if __name__ == '__main__':
    input_file = "lena_mono.bmp"
    output1 = "lena_filter_04.bmp"
    output2 = "lena_filter_05.bmp"
    output3 = "lena_filter_06.bmp"
    print("h[3,3]")
    highpass_mean(input_file, output1, 3)
    print("h[5,5]")
    highpass_mean(input_file, output2, 5)
    print("h[7,7]")
    highpass_mean(input_file, output3, 7)

