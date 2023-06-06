import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap
import torch
from skimage.color import label2rgb
import cv2
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision.utils import draw_bounding_boxes
from extract import extract_utils as utils


def plot_segmentation(
    images_list: str,
    images_root: str,
    segmentations_dir: str,
    bbox_file: str = None,
    output_dir: str = "./output_plots/segm"
):
    utils.make_output_dir(output_dir, check_if_empty=False)

    # Inputs
    image_paths = []
    segmap_paths = []
    images_root = Path(images_root)
    segmentations_dir = Path(segmentations_dir)
    for image_file in Path(images_list).read_text().splitlines():
        segmap_file = f'{Path(image_file).stem}.png'
        image_paths.append(images_root / image_file)
        segmap_paths.append(segmentations_dir / segmap_file)
    print(f'Found {len(image_paths)} image and segmap paths')

    # Load optional bounding boxes
    if bbox_file is not None:
        bboxes_list = torch.load(bbox_file)

    # Colors
    colors = get_cmap('tab20', 21).colors[:, :3]

    # Load
    for i, (image_path, segmap_path) in enumerate(zip(image_paths, segmap_paths)):
        image_id = image_path.stem
        
        # Load
        image = np.array(Image.open(image_path).convert('RGB'))
        segmap = np.array(Image.open(segmap_path))

        # Convert binary
        if set(np.unique(segmap).tolist()) == {0, 255}:
            segmap[segmap == 255] = 1

        # Resize
        segmap_fullres = cv2.resize(segmap, dsize=image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

        # Only view images with a specific class
        # which_index = 1
        # if which_index not in np.unique(segmap):
            # continue

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))

        # Load optional bounding boxes
        bboxes = None
        if bbox_file is not None:
            bboxes = torch.tensor(bboxes_list[i]['bboxes_original_resolution'])
            assert bboxes_list[i]['id'] == image_id, f"{bboxes_list[i]['id']} but {image_id}"
            image_torch = torch.from_numpy(image).permute(2, 0, 1)
            image_with_boxes_torch = draw_bounding_boxes(image_torch, bboxes)
            image_with_boxes = image_with_boxes_torch.permute(1, 2, 0).numpy()
            
            axes[0].imshow(image_with_boxes)
            axes[0].set_title('Image with Bounding Boxes')
            axes[0].axis('off')
        else:
            axes[0].imshow(image)
            axes[0].set_title('Image')
            axes[0].axis('off')
        

        # Color
        segmap_label_indices, segmap_label_counts = np.unique(segmap, return_counts=True)
        blank_segmap_overlay = label2rgb(label=segmap_fullres, image=np.full_like(image, 128), 
            colors=colors[segmap_label_indices[segmap_label_indices != 0]], bg_label=0, alpha=1.0)
        image_segmap_overlay = label2rgb(label=segmap_fullres, image=image, 
            colors=colors[segmap_label_indices[segmap_label_indices != 0]], bg_label=0, alpha=0.45)
        segmap_caption = dict(zip(segmap_label_indices.tolist(), (segmap_label_counts).tolist()))

        # Visualization of blank segmap overlay
        axes[1].imshow(blank_segmap_overlay)
        axes[1].set_title('Blank Segmentation Overlay')
        axes[1].axis('off')

        # Visualization of colored image
        axes[2].imshow(image_segmap_overlay)
        axes[2].set_title('Image with Segmentation Overlay')
        axes[2].axis('off')

        # Save the plot
        output_filename = os.path.join(output_dir, f"{image_id}.png")
        plt.savefig(output_filename)
        # plt.close(fig)

    print(f"Plots saved in the output directory: {output_dir}")

