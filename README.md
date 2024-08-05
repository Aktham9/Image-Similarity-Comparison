# Image-Similarity-Comparison
This repository contains a Python script to compare two images based on the spatial distribution of black pixels. The script uses Structural Similarity Index (SSIM) and a pixel count difference tolerance to determine if two images are similar. The comparison excludes exact identical images.

## Features

- Converts images to black and white (binary) format.
- Converts images to grayscale to perform SSIM calculation.
- Uses SSIM to measure the spatial similarity of black pixels.
- Counts the number of black pixels in each image.
- Considers images similar if the SSIM index is high but less than 1, and the pixel difference is within a specified tolerance but greater than 0.

## Requirements

- Python 3.x
- Pillow (PIL)
- scikit-image

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/SSIM-Image-Comparison.git
    cd SSIM-Image-Comparison
    ```

2. Install the required packages:
    ```sh
    pip install pillow scikit-image
    ```

## Usage

1. Place your images in the repository directory.

2. Update the `image_path1` and `image_path2` variables in the script with the paths to your images.

3. Run the script:
    ```sh
    python compare_images.py
    ```

## Example

Here is an example of how the script works:

- **Total pixels in image 1:** 1,048,576
- **Black pixels in image 1:** 251,459
- **Total pixels in image 2:** 1,048,576
- **Black pixels in image 2:** 251,778
- **Difference in black pixels:** 319
- **SSIM index:** 0.9867

Based on these values, the script will determine if the images are similar based on the criteria.

## Script Overview

```python
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

def convert_to_black_and_white(image_path):
    """
    Convert an image to black and white, where black pixels remain black
    and all other pixels become white.
    """
    image = Image.open(image_path).convert('RGB')
    np_image = np.array(image)

    # Create a mask for black pixels
    black_mask = (np_image[:, :, 0] == 0) & (np_image[:, :, 1] == 0) & (np_image[:, :, 2] == 0)
    
    # Create a new image with white background and black pixels
    bw_image = np.ones_like(np_image) * 255  # Start with a white image
    bw_image[black_mask] = [0, 0, 0]  # Set black pixels to black

    return Image.fromarray(bw_image)

def count_black_pixels(image):
    """
    Count the number of black pixels in a grayscale image.
    """
    np_image = np.array(image.convert('L'))  # Convert image to grayscale
    return np.sum(np_image == 0)

def compare_images(image_path1, image_path2, tolerance, ssim_threshold):
    """
    Compare two images based on the SSIM index and pixel count difference.
    """
    bw_image1 = convert_to_black_and_white(image_path1)
    bw_image2 = convert_to_black_and_white(image_path2)
    
    black_pixels_image1 = count_black_pixels(bw_image1)
    black_pixels_image2 = count_black_pixels(bw_image2)
    
    pixel_difference = abs(black_pixels_image1 - black_pixels_image2)
    
    # Convert images to grayscale (L mode) and then to numpy arrays for SSIM calculation
    np_bw_image1 = np.array(bw_image1.convert('L'))
    np_bw_image2 = np.array(bw_image2.convert('L'))

    # Calculate SSIM index for the grayscale images
    ssim_index, _ = ssim(np_bw_image1, np_bw_image2, full=True)

    print(f"Total pixels in image 1: {np_bw_image1.size}")
    print(f"Black pixels in image 1: {black_pixels_image1}")
    print(f"Total pixels in image 2: {np_bw_image2.size}")
    print(f"Black pixels in image 2: {black_pixels_image2}")
    print(f"Difference in black pixels: {pixel_difference}")
    print(f"SSIM index: {ssim_index}")

    # Images are considered similar if SSIM index is high but less than 1, pixel difference is within tolerance but greater than 0
    return 0 < pixel_difference <= tolerance and ssim_threshold < ssim_index < 1

# File paths to the uploaded images
image_path1 = 'path_to_your_first_image.jpg'
image_path2 = 'path_to_your_second_image.jpg'

# Define a tolerance level for the difference in black pixels and SSIM threshold
tolerance = 100  # Adjust this value as needed
ssim_threshold = 0.95  # Adjust this value as needed

# Check if images are similar
if compare_images(image_path1, image_path2, tolerance, ssim_threshold):
    print("The images are similar.")
else:
    print("The images are not similar.")
