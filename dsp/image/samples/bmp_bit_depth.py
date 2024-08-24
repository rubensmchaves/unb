from PIL import Image

# Load the BMP image
image = Image.open('lena_mono.bmp')

# Check if the image is a BMP file
if image.format == 'BMP':
    # Determine the bit depth based on the mode
    mode = image.mode
    
    if mode == '1':  # 1-bit pixels, black and white
        bit_depth = 1
    elif mode == 'L':  # 8-bit pixels, grayscale
        bit_depth = 8
    elif mode == 'P':  # 8-bit pixels, color palette
        bit_depth = 8
    elif mode == 'RGB':  # 8-bit pixels, true color
        bit_depth = 24
    elif mode == 'RGBA':  # 8-bit pixels, true color with transparency
        bit_depth = 32
    elif mode == 'RGBX':  # 32-bit pixels, true color with padding
        bit_depth = 32
    else:
        bit_depth = None  # Unsupported mode

    if bit_depth is not None:
        print(f"Image mode: {mode}")
        print(f"Bit depth: {bit_depth} bits per pixel")
    else:
        print("Could not determine the bit depth for this mode.")
else:
    print("The file is not a BMP image.")
