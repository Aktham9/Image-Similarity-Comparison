from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

def convert_to_black_and_white(image_path):
    image = Image.open(image_path).convert('RGB')
    np_image = np.array(image)

    # Create a mask for black pixels
    black_mask = (np_image[:, :, 0] == 0) & (np_image[:, :, 1] == 0) & (np_image[:, :, 2] == 0)
    
    # Create a new image with white background and black pixels
    bw_image = np.ones_like(np_image) * 255  # Start with a white image
    bw_image[black_mask] = [0, 0, 0]  # Set black pixels to black

    return Image.fromarray(bw_image)

def count_black_pixels(image):
    np_image = np.array(image.convert('L'))  # Convert image to grayscale
    return np.sum(np_image == 0)

def get_total_pixels(image):
    np_image = np.array(image.convert('L'))  # Convert image to grayscale
    return np_image.size

def compare_images(image_path1, image_path2, tolerance, ssim_threshold):
    bw_image1 = convert_to_black_and_white(image_path1)
    bw_image2 = convert_to_black_and_white(image_path2)
    
    total_pixels_image1 = get_total_pixels(bw_image1)
    total_pixels_image2 = get_total_pixels(bw_image2)
    
    black_pixels_image1 = count_black_pixels(bw_image1)
    black_pixels_image2 = count_black_pixels(bw_image2)
    
    pixel_difference = abs(black_pixels_image1 - black_pixels_image2)
    ssim_index, _ = ssim(np.array(bw_image1.convert('L')), np.array(bw_image2.convert('L')), full=True)

    print(f"Total pixels in image 1: {total_pixels_image1}")
    print(f"Black pixels in image 1: {black_pixels_image1}")
    print(f"Total pixels in image 2: {total_pixels_image2}")
    print(f"Black pixels in image 2: {black_pixels_image2}")
    print(f"Difference in black pixels: {pixel_difference}")
    print(f"SSIM index: {ssim_index}")

    #return ssim_index > ssim_threshold and pixel_difference <= tolerance
    return ssim_index > ssim_threshold and 0 < pixel_difference <= tolerance

# File paths to the uploaded images
image_path1 = ''
image_path2 =''
# Define a tolerance level for the difference in black pixels and SSIM threshold
tolerance = 400  # Adjust this value as needed
ssim_threshold = 0.95  # Adjust this value as needed

# Check if images are similar
if compare_images(image_path1, image_path2, tolerance, ssim_threshold):
    print("The images are similar.")
else:
    print("The images are not similar.")
