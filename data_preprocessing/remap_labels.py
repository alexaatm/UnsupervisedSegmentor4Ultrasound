import os
from PIL import Image
import numpy as np
import argparse

def remap_lables(old_label_image, old_labels, new_labels):
    old_labels_unique = np.unique(old_label_image)
    print('old_current', old_labels_unique)
    print('old_assumed', old_labels)
    # for old_current, old_assumed in zip(old_labels_unique,np.array(old_labels) ):
    #     assert(old_current==old_assumed)
    new_label_image = np.zeros_like(old_label_image)
    for old_label_i, target_label_i in zip(old_labels, new_labels):
        new_label_image[old_label_image == int(old_label_i)] = int(target_label_i)
    return new_label_image

def remap_labels_in_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through images in input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # Load PNG image
            image_path = os.path.join(input_folder, filename)
            im = np.array(Image.open(image_path))

            # get current and new labels
            current_labels = np.unique(im).tolist()
            new_labels = [i for i in range(len(current_labels))]

            # remap
            new_im = remap_lables(im, current_labels, new_labels)

            # save
            new_segm_image = Image.fromarray(new_im)
            output_path = os.path.join(output_folder, filename)
            new_segm_image.save(str(output_path))

def main():
    parser = argparse.ArgumentParser(description="Remap labels in a folder.")
    parser.add_argument("input_folder", help="Path to the input folder containing PNG images.")
    parser.add_argument("output_folder", help="Path to the output folder for saving normalized images.")
    args = parser.parse_args()

    remap_labels_in_folder(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
