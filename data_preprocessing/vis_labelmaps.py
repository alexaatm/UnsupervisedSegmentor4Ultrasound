import os
from PIL import Image
import argparse

def create_images_from_labelmaps(folder_path, output_path):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get all PNG files in the folder
    labelmap_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    for labelmap_file in labelmap_files:
        print(f'Processing file: {labelmap_file}')
        labelmap_path = os.path.join(folder_path, labelmap_file)
        image_name = os.path.splitext(labelmap_file)[0] + '_converted.png'
        image_path = os.path.join(output_path, image_name)

        # Open the label map image
        labelmap = Image.open(labelmap_path)

        # Create a new image with RGB mode
        image = Image.new("RGB", labelmap.size, color="#FFFFFF")

       # Map class labels to colors and set pixels
        for y in range(labelmap.size[1]):
            for x in range(labelmap.size[0]):
                label = labelmap.getpixel((x, y))  # Directly use the pixel value
                if label == 0:
                    color = "#98b8eb"
                elif label ==1:
                    color = "#e89494"
                elif label ==2:
                    color= "#90c29f"
                elif label ==3:
                    color == "#f1b0c6"
                else:
                    color = "#eeac89"
                
                image.putpixel((x, y), tuple(int(color[i:i + 2], 16) for i in (1, 3, 5)))

        # Save the new image
        image.save(image_path)
        print(f"Image saved: {image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert label maps to images")
    parser.add_argument("--folder_path", help="Path to the folder containing label map images")
    parser.add_argument("--output_path", help="Path to the output folder for converted images")
    args = parser.parse_args()

    create_images_from_labelmaps(args.folder_path, args.output_path)
