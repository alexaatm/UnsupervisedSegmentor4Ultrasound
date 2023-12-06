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
from torchvision import transforms
from torch import nn

torch.cuda.empty_cache()
from torch.cuda.amp import autocast

import argparse

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

def plot_eigenvectors(
    images_list: str,
    images_root: str,
    eigenvevtors_dir: str,
    features_dir: str,
    output_dir: str = "./output_plots/eigen"
):
    utils.make_output_dir(output_dir, check_if_empty=False)

    # Inputs
    image_paths = []
    eigen_paths = []
    feat_paths = []
    images_root = Path(images_root)
    eigenvevtors_dir = Path(eigenvevtors_dir)
    features_dir = Path(features_dir)
    for image_file in Path(images_list).read_text().splitlines():
        file = f'{Path(image_file).stem}.pth'
        image_paths.append(images_root / image_file)
        eigen_paths.append(eigenvevtors_dir / file)
        feat_paths.append(features_dir / file)
    print(f'Found {len(image_paths)} image and eigen paths')

    # Load
    for i, (image_path, feat_path, eigen_path) in enumerate(zip(image_paths, feat_paths, eigen_paths)):
        image_id = image_path.stem
        print(image_id)
        
        # Load data dictionary
        image = np.array(Image.open(image_path).convert('RGB'))
        data_dict = torch.load(feat_path, map_location='cpu')
        data_dict.update(torch.load(eigen_path, map_location='cpu'))
        eigenvec_num = len(data_dict['eigenvectors'])
        eigenvectors = data_dict['eigenvectors'][:eigenvec_num].numpy()
        # print(eigenvectors.shape)

        # Reshape eigenvevtors
        B, C, H, W, P, H_patch, W_patch, H_pad, W_pad = utils.get_image_sizes(data_dict)
        eigenvectors_img=eigenvectors.reshape(eigenvec_num, H_patch, W_patch)

        # Plot
        fig, axes = plt.subplots(nrows=2, ncols=eigenvec_num//2+1, figsize=(15, 5))
        for i, eigv_ax_pair in enumerate(zip(axes.flatten(),eigenvectors_img)):
            a, eigv = eigv_ax_pair
            a.imshow(eigv)
            a.title.set_text("eigv "+str(i))

        for a in axes.flatten(): 
            a.axis('off')

        plt.tight_layout()

        # Save the plot
        output_filename = os.path.join(output_dir, f"{image_id}_eigenvectors.png")
        plt.savefig(output_filename)

    print(f"Plots saved in the output directory: {output_dir}")

def plot_dino_attn_maps(
    images_list: str,
    images_root: str,
    model_name: str,
    model_checkpoint: str = "",
    output_dir: str = "./output_plots/dino_attn_maps"
):
    utils.make_output_dir(output_dir, check_if_empty=False)


    # Inputs
    image_paths = []
    images_root = Path(images_root)
    for image_file in Path(images_list).read_text().splitlines():
        image_paths.append(images_root / image_file)
    print(f'Found {len(image_paths)} image paths')

    
    # Get the model
    if model_checkpoint=="" or model_checkpoint==None:
        model, val_transform, patch_size, nh = utils.get_model(model_name)
    else:
        model, val_transform, patch_size,  nh = utils.get_model_from_checkpoint(model_name, model_checkpoint, just_backbone=True)

    # disable grad
    for p in model.parameters():
        p.requires_grad = False

    # put model to cuda device to
    model = model.to('cuda')
    
    # Define transforms
    # TODO: check with val_transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    # Load
    for i, image_path in enumerate(image_paths):
        image_id = image_path.stem
        print(image_id)
        
        # Load image - sample for processing, input_img for plotting the original image
        sample = np.array(Image.open(image_path).convert('RGB'))
        input_img = sample 
        # Convert PIL Image to NumPy array and transpose dimensions
        input_img = np.array(input_img).transpose((2, 0, 1))  # Transpose to (channels, height, width)

        # Apply transform
        sample = transform(sample)
        # print(f'sample.shape={sample.shape}')

        # Plot
        w = sample.shape[1] - sample.shape[1] % patch_size
        h = sample.shape[2] - sample.shape[2] % patch_size
        sample = sample[:, :w, :h].unsqueeze(0)
        w_featmap = sample.shape[-2] // patch_size
        h_featmap = sample.shape[-1] // patch_size

        # move image to device
        sample = sample.to('cuda')

        # get self-attention
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                torch.cuda.empty_cache()
                attentions = model.get_last_selfattention(sample)
            
        # we keep only the output patch attention
        if 'dinov2' in model_name:
            # in dinov2, attentions return tensor with 3 dimensions, if xformers is enabled (make sure export XFORMERS_DISABLED=True)
            # If xformers is disabled, the commented code below is not needed
            # attentions = torch.unsqueeze(attentions, 1)
            # attentions.fill_(nh)
            # print(f'attentions.shape={attentions.shape}')
            if 'reg' in model_name:
                attentions = attentions[0, :, 0, 1+4:].reshape(nh, -1)
            else: 
                attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
        else:
            attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)

        threshold = 0.6 # We visualize masks obtained by thresholding the self-attention maps to keep xx% of the mass.
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()

        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().detach().numpy()
        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().detach().numpy()
        attentions_mean = np.mean(attentions, axis=0)

        fig = plt.figure(figsize=(6, 6), dpi=200)
        ax = fig.add_subplot(3, 3, 1)
        ax.set_title("Input")
        ax.imshow(np.transpose(input_img, (1, 2, 0)))
        ax.axis("off")

        # visualize self-attention of each head
        for i in range(6):
            ax = fig.add_subplot(3, 3, i + 4)
            ax.set_title("Head " + str(i + 1))
            ax.imshow(attentions[i])
            ax.axis("off")

        ax = fig.add_subplot(3, 3, 2)
        ax.set_title("Head Mean")
        ax.imshow(attentions_mean)
        ax.axis("off")

        fig.tight_layout()

        # Save the plot
        output_filename = os.path.join(output_dir, f"{image_id}_{model_name}_attn_maps.png")
        fig.savefig(output_filename)

        # Close plot
        plt.close()

    print(f"Plots saved in the output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot DINO Attention Maps')
    parser.add_argument('--images_list', type=str, required=True, help='Path to the file containing the list of image filenames')
    parser.add_argument('--images_root', type=str, required=True, help='Root directory of the images')
    parser.add_argument('--model_checkpoint', type=str, required=False, help='Path to the DINO model checkpoint')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the DINO model')
    parser.add_argument('--output_dir', type=str, default='./output_plots/dino_attn_maps', help='Output directory for saving plots')

    args = parser.parse_args()

    # Print GPU memory summary
    print(torch.cuda.memory_summary())  

    # Call the function with command-line arguments
    plot_dino_attn_maps(images_list=args.images_list, 
                        images_root=args.images_root, 
                        model_checkpoint=args.model_checkpoint,
                        model_name=args.model_name,
                        output_dir=args.output_dir)

