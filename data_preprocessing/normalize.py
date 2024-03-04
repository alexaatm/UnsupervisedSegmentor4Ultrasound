import os
from PIL import Image
import numpy as np
import argparse

def normalize_images_in_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through images in input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # Load PNG image
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            # Convert image to numerical array
            image_array = np.array(image)

            # Find max value
            max_value = np.max(image_array)

            # Normalize values to range 0-255
            normalized_array = (image_array / max_value) * 255

            # Round normalized values and convert to uint8
            normalized_array = np.round(normalized_array).astype(np.uint8)

            # Convert array back to image
            normalized_image = Image.fromarray(normalized_array)

            # Save normalized image to output folder
            output_path = os.path.join(output_folder, filename)
            normalized_image.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="Normalize images in a folder.")
    parser.add_argument("input_folder", help="Path to the input folder containing PNG images.")
    parser.add_argument("output_folder", help="Path to the output folder for saving normalized images.")
    args = parser.parse_args()

    normalize_images_in_folder(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
