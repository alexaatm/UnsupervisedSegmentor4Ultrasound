import os
from PIL import Image
import numpy as np
import argparse
from skimage import io
import matplotlib.pyplot as plt

# from skimage.data import lena
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float

def generate_fz(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through images in input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # Load PNG image
            image_path = os.path.join(input_folder, filename)
            img = io.imread(image_path, as_gray=False, plugin='matplotlib')
            img = img_as_float(img)
            # segmap = slic(img, n_segments=15, compactness=0.1, min_size_factor=0.01, sigma=0.1, channel_axis=None)
            segmap = felzenszwalb(img, sigma=0.1, min_size=1000)

            # Round normalized values and convert to uint8
            segmap = np.round(segmap).astype(np.uint8)

            # Convert array back to image
            segmap = Image.fromarray(segmap)

            # Save normalized image to output folder
            output_path = os.path.join(output_folder, filename)
            segmap.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="Generate felzenszwalb segmentation in a folder.")
    parser.add_argument("input_folder", help="Path to the input folder containing PNG images.")
    parser.add_argument("output_folder", help="Path to the output folder for saving normalized images.")
    args = parser.parse_args()

    generate_fz(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
