import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import kornia as K
import torch

from torch.nn import Module
from torch import Tensor
from typing import Any, Callable, Iterable, Optional, Tuple, Union
from kornia.enhance import equalize_clahe

import argparse


class EqualizeClahe(Module):
    def __init__(self, 
                clip_limit: float = 40.0,
                grid_size: Tuple[int, int] = (8, 8),
                slow_and_differentiable: bool = False
                 ) -> None:
        super().__init__()
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        self.slow_and_differentiable = slow_and_differentiable

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(clip_limit={self.clip_limit}, "
            f"grid_size={self.grid_size}, "
            f"slow_and_differentiable={self.slow_and_differentiable})"
        )

    def forward(self, input: Tensor) -> Tensor:
        # ref: https://kornia.readthedocs.io/en/latest/_modules/kornia/enhance/equalization.html#equalize_clahe
        return equalize_clahe(input, self.clip_limit, self.grid_size, self.slow_and_differentiable)

def preprocess_dataset(image_folder, processed_image_folder, preprocessing_pipeline):
    # preprocessing_pipeline: of type torch.transforms or K.augmentation.container.ImageSequential

    if not os.path.exists(processed_image_folder):
        os.makedirs(processed_image_folder)

    images = sorted(os.listdir(image_folder))

    for im_name in tqdm(images):
        # read file
        im_file = os.path.join(image_folder, im_name)
        x_rgb: torch.Tensor = K.io.load_image(im_file, K.io.ImageLoadType.RGB32)[None, ...]  # BxCxHxW
        
        # process
        x = preprocessing_pipeline(x_rgb)
        x_numpy = K.utils.image.tensor_to_image(x)
        processed_image = Image.fromarray(np.uint8(x_numpy* 255))

        # save
        new_im_file = os.path.join(processed_image_folder, im_name)
        
        processed_image.save(str(new_im_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing of ultrasound data (Offline)')
    parser.add_argument('--dataset_folder', type=str, 
                        default='/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/CAROTID_MIXED/val_mini',
                        help='Path to the root dataset folder containing folder "images"')
    args = parser.parse_args()

    preprocessing_pipeline = K.augmentation.container.ImageSequential(
        K.filters.GaussianBlur2d((3,3), (5.,5.)),
        EqualizeClahe(grid_size = (2,2)),
        # K.filters.MedianBlur((5,5)),
    )
    
    folders_to_process = [
        # TODO: account for passing a folder with folders, or a list of folder paths
        args.dataset_folder,
    ]

    for root_folder in folders_to_process:
        images_folder = os.path.join(root_folder, 'images')
        assert(os.path.exists(images_folder))
        preprocessed_folder = os.path.join(root_folder, 'preprocessed')
        preprocess_dataset(images_folder,preprocessed_folder,preprocessing_pipeline)
