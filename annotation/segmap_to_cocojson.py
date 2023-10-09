import os
import argparse
import json
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

def get_coco_color_map(num_colors):
    cmap = plt.get_cmap('tab20', num_colors)
    rgb_colors = (cmap(np.arange(num_colors))[:, :3] * 255).astype(int)
    normalized_colors = (rgb_colors / 255.0).tolist()
    return normalized_colors

def segmaps_to_coco(image_folder, segmap_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of image filenames from the image folder
    image_filenames = os.listdir(image_folder)
    
    # Initialize the COCO-style annotations dictionary
    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Initialize the category ID counter
    category_id = 1

    # Get the COCO color map
    coco_colors = get_coco_color_map(21)

    for idx, image_filename in enumerate(image_filenames):
        # Create a new entry for each image in the COCO-style annotations
        image_entry = {
            "id": idx + 1,
            "file_name": image_filename,
            "width": 0,  # Replace with the actual image width
            "height": 0,  # Replace with the actual image height
        }
        
        # Read the segmentation map for the corresponding image
        segmap_filename = os.path.join(segmap_folder, image_filename)
        segmap_filename = segmap_filename.replace(".jpg", ".png")
        segmap = np.array(Image.open(segmap_filename))
        
        # Extract unique category labels from the segmentation map
        unique_categories = np.unique(segmap)
        
        for category_label in unique_categories:
            # Skip background category (usually label 0)
            if category_label == 0:
                continue
            
            # Create a new category entry
            category_entry = {
                "id": category_id,
                "name": f"category_{category_label}",  # Replace with actual category names if available
                "color": coco_colors[category_id - 1],  # Use the assigned color for the category
                "isthing": 1,
                "supercategory": "object",
            }
            
            coco_annotations["categories"].append(category_entry)
            
            # Find the coordinates of pixels with the current category label
            category_mask = (segmap == category_label).astype(np.uint8)
            # Make the mask Fortran contiguous
            category_mask = np.asfortranarray(category_mask)
            # Encode the segmentation mask using RLE (Run-Length Encoding)
            category_mask_encoded = mask_util.encode(category_mask)
            category_mask_encoded["counts"] = category_mask_encoded["counts"].decode("utf-8")  # Convert bytes to string
            
            # Convert the segmentation mask to a list if it's not already
            if not isinstance(category_mask_encoded, list):
                category_mask_encoded = [category_mask_encoded]

            # Create an annotation entry for the current category
            annotation_entry = {
                "id": len(coco_annotations["annotations"]) + 1,
                "image_id": image_entry["id"],
                "category_id": category_id,
                "segmentation": category_mask_encoded,
                "area": int(np.sum(category_mask)),
                "bbox": mask_util.toBbox(category_mask_encoded).tolist(),
                "iscrowd": 0,
            }
            
            coco_annotations["annotations"].append(annotation_entry)
            
            # Increment the category ID counter
            category_id += 1
        
        # Append the image entry to the list of images
        coco_annotations["images"].append(image_entry)
    
    # Save the COCO-style annotations to a JSON file
    output_filename = os.path.join(output_folder, "annotations.json")
    with open(output_filename, "w") as f:
        json.dump(coco_annotations, f)

if __name__ == "__main__":
    # Set up argparse to parse command line arguments
    parser = argparse.ArgumentParser(description="Convert segmentation maps to COCO-style annotations.")
    parser.add_argument("--image_folder", type=str, help="Path to the folder containing input images.")
    parser.add_argument("--segmap_folder", type=str, help="Path to the folder containing segmentation maps.")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder for saving COCO annotations.")
    args = parser.parse_args()

    # Call the function to convert segmaps to COCO-style annotations
    segmaps_to_coco(args.image_folder, args.segmap_folder, args.output_folder)
