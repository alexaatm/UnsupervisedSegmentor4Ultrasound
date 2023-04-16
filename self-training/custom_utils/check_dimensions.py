import sys
from PIL import Image
import os

def check_image_dimensions(path, expected_size):
    # get the list of files in the directory
    files = os.listdir(path)
    
    # initialize a list to store filenames that do not have the expected dimensions
    non_matching_files = []
    
    # loop through each file in the directory
    for file in files:
        # check if the file is an image file (i.e., has a file extension of .jpg, .png, etc.)
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            # open the image file using the Pillow library
            with Image.open(os.path.join(path, file)) as img:
                # get the width and height of the image
                width, height = img.size
                # check if the dimensions of the image match the expected dimensions
                if (width, height) != expected_size:
                    non_matching_files.append(file)
                    print(f'File {file} has size: ({width},{height})')
    
    # return the list of filenames that do not have the expected dimensions
    return non_matching_files

def main():
    # check if the correct number of arguments were provided
    if len(sys.argv) != 4:
        print("Usage: python script.py path height width")
        return
    
    # get the path, height, and width from the command line arguments
    path = sys.argv[1]
    height = int(sys.argv[2])
    width = int(sys.argv[3])
    
    # check if the specified directory exists
    if not os.path.exists(path):
        print(f"The directory {path} does not exist")
        return
    
    # check the dimensions of the images in the directory
    non_matching_files = check_image_dimensions(path, (width, height))
    
    # print the filenames of the images that do not have the expected dimensions
    if non_matching_files:
        print("The following files do not have the expected dimensions:")
        for file in non_matching_files:
            print(file)
    else:
        print("All image files in the directory have the expected dimensions")

if __name__ == '__main__':
    main()
