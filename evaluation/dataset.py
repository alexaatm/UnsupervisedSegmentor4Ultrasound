import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class EvalDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.gt_dir = os.path.join(root_dir, 'ground_truth')
        self.pred_dir = os.path.join(root_dir, 'predictions')
        self.image_list = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name_full = self.image_list[idx]
        img_name = img_name_full[:-4]
        img_path = os.path.join(self.image_dir, img_name_full)
        gt_path = os.path.join(self.gt_dir, img_name + ".png")
        pred_path = os.path.join(self.pred_dir, img_name + ".png")

        image = np.array(Image.open(img_path).convert("RGB"))
        ground_truth = np.array(Image.open(gt_path).convert('L'))
        prediction = np.array(Image.open(pred_path).convert('L'))

        metadata = {'id': Path(img_path).stem, 'path': img_path, 'shape': tuple(image.shape[:2])}

        return (image, ground_truth, prediction, metadata)