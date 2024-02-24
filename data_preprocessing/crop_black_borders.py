import os
from PIL import Image
import numpy as np
from tqdm import tqdm

import argparse

def crop_image(image, top, left, bottom, right):
    width, height = image.size
    left = max(0, left)
    top = max(0, top)
    right = min(width, width - right)
    bottom = min(height, height - bottom)

    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

def find_distance_to_non_zero(image_array):
    top, left, bottom, right = 0, 0, 0, 0

    # Find distance from top
    for i in range(image_array.shape[0]):
        if np.any(image_array[i]):
            top = i
            break

    # Find distance from bottom
    for i in range(image_array.shape[0] - 1, -1, -1):
        if np.any(image_array[i]):
            bottom = image_array.shape[0] - i - 1
            break

    # Find distance from left
    for i in range(image_array.shape[1]):
        if np.any(image_array[:, i]):
            left = i
            break

    # Find distance from right
    for i in range(image_array.shape[1] - 1, -1, -1):
        if np.any(image_array[:, i]):
            right = image_array.shape[1] - i - 1
            break

    return top, left, bottom, right

def preprocess_and_find_distances(image_folder, images):
    distances = {'top': [], 'left': [], 'bottom': [], 'right': []}

    for im_name in tqdm(images):
        # read file
        im_file = os.path.join(image_folder, im_name)
        image = Image.open(im_file)
        image_array = np.array(image)

        # find distances
        top, left, bottom, right = find_distance_to_non_zero(image_array)

        distances['top'].append(top)
        distances['left'].append(left)
        distances['bottom'].append(bottom)
        distances['right'].append(right)

    min_top = min(distances['top'])
    min_left = min(distances['left'])
    min_bottom = min(distances['bottom'])
    min_right = min(distances['right'])

    return min_top, min_left, min_bottom, min_right

def crop_and_save_images(image_folder, processed_image_folder, images, distances):
    if not os.path.exists(processed_image_folder):
        os.makedirs(processed_image_folder)

    min_top, min_left, min_bottom, min_right  = distances

    for im_name in tqdm(images):
        # read file
        im_file = os.path.join(image_folder, im_name)
        image = Image.open(im_file)

        # crop image
        cropped_image = crop_image(image, min_top, min_left, min_bottom, min_right)

        # save cropped image
        new_im_file = os.path.join(processed_image_folder, im_name)
        cropped_image.save(new_im_file)

        # print(f"Processed {im_name}, Cropped and Saved.")
    
def split_images_into_sequences(images):
    sequences = {}
    for im_name in images:
        prefix = im_name.split('_')[0] + '_'
        if prefix not in sequences:
            sequences[prefix] = []
        sequences[prefix].append(im_name)
    return sequences

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cutting black borders of ultrasound data (Offline)')
    parser.add_argument('--image_folder', type=str, 
                        help='Path to the root dataset folder containing folder "images"')
    parser.add_argument('--additional_folders', nargs='+', default=[],
                        help='List of additional folders to be cropped the same way as the main folder')
    
    args = parser.parse_args()

    processed_image_folder = args.image_folder + "_cropped"

    images = sorted(os.listdir(args.image_folder))

    # Split all images into sequences based on the prefix
    image_sequences = split_images_into_sequences(images)

    # Process each sequence separately
    for sequence_prefix, sequence_images in image_sequences.items():
        print(f'Processing image folder for sequence {sequence_prefix}')
        distances = preprocess_and_find_distances(args.image_folder, sequence_images)
        crop_and_save_images(args.image_folder, processed_image_folder, sequence_images, distances)

        # Crop images in additional folders
        if args.additional_folders:
            print(f'Processing additional folders for sequence {sequence_prefix}')

            for folder in args.additional_folders:
                processed_image_folder = folder + "_cropped"
                crop_and_save_images(folder, processed_image_folder, sequence_images, distances)

