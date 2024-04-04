import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm, trange
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, gt_dir = "", pred_dir = "", transform: object = None, image_dir = "", label_map = None):
        self.root_dir = root_dir
        self.image_dir = image_dir if image_dir!="" else os.path.join(root_dir, 'images')
        self.pred_dir = pred_dir if pred_dir!="" else os.path.join(root_dir, 'predictions')
        self.image_list = os.listdir(self.image_dir)
        self.transform = transform
        self._prepare_label_map(label_map)

        # Use pseudolabels as Ground Truth is no GT directory given (e.g. for self training)
        if gt_dir is not None:
            self.gt_dir = gt_dir if gt_dir!="" else os.path.join(root_dir, 'ground_truth')
        else:
            self.gt_dir = self.pred_dir #HACK, cause gt is ignored during training

        print("root:", self.root_dir)
        print("image_dir:", self.image_dir)
        print("gt_dir:", self.gt_dir)
        print("pred_dir:", self.pred_dir)
        print("label_map: ", label_map)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name_full = self.image_list[idx]
        img_name = img_name_full[:-4]
        img_path = os.path.join(self.image_dir, img_name_full)
        gt_path = os.path.join(self.gt_dir, img_name + ".png")
        pred_path = os.path.join(self.pred_dir, img_name + ".png")

        # assert(os.path.isfile(gt_path))

        # Load data
        image = np.array(Image.open(img_path).convert("RGB"))
        ground_truth = np.array(Image.open(gt_path).convert('L'))
        prediction = np.array(Image.open(pred_path).convert('L'))
        metadata = {'id': Path(img_path).stem, 'path': img_path, 'shape': tuple(image.shape[:2])}

        # Resize masks of image size not matching
        prediction = self._resize_mask(prediction, image)
        ground_truth = self._resize_mask(ground_truth, image)

        # Remap labelmap if matching provided
        if self.label_map_fn is not None:
            prediction = self.label_map_fn(prediction)

        # Tranform and unpack
        if self.transform is not None:
            if type(self.transform) is tuple:
                for t in self.transform:
                    data = t(image=image, mask1=ground_truth, mask2=prediction)
                    image, ground_truth, prediction = data['image'], data['mask1'], data['mask2']

        if torch.is_tensor(ground_truth):
            ground_truth = ground_truth.long()
        if torch.is_tensor(prediction):
            prediction = prediction.long()

        return image, ground_truth, prediction, metadata
    
    def _resize_mask(self, mask, image):
        # Check if sizes correspond
        H_im, W_im = image.shape[:2]
        H_mask, W_mask = mask.shape

        if (H_mask!= H_im or W_mask!=W_im):
            mask = cv2.resize(mask, dsize=(H_im, W_im), interpolation=cv2.INTER_NEAREST)  # (H, W)
        return mask

    def _prepare_label_map(self, label_map):
        if label_map is not None:
            self.label_map_fn = np.vectorize(label_map.__getitem__)
        else:
            self.label_map_fn = None
            