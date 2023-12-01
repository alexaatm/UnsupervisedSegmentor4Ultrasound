import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm, trange
from torchvision import transforms


class EvalDataset(Dataset):
    def __init__(self, root_dir, gt_dir = "", pred_dir = "", transform: object = None, image_dir = ""):
        self.root_dir = root_dir
        self.image_dir = image_dir if image_dir!="" else os.path.join(root_dir, 'images')
        self.gt_dir = gt_dir if gt_dir!="" else os.path.join(root_dir, 'ground_truth')
        self.pred_dir = pred_dir if pred_dir!="" else os.path.join(root_dir, 'predictions')
        self.image_list = os.listdir(self.image_dir)
        self.transform = transform

        print("root:", self.root_dir)
        print("image_dir:", self.image_dir)
        print("gt_dir:", self.gt_dir)
        print("pred_dir:", self.pred_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name_full = self.image_list[idx]
        img_name = img_name_full[:-4]
        img_path = os.path.join(self.image_dir, img_name_full)
        gt_path = os.path.join(self.gt_dir, img_name + ".png")
        pred_path = os.path.join(self.pred_dir, img_name + ".png")

        # assert(os.path.isfile(gt_path))

        if self.transform is not None:
            image = self.transform(Image.open(img_path).convert("RGB"))
            # print(f"EvalDataset: Image after transforms: {image.size()}")
            image = image.detach()
            if image.is_cuda:
                image = image.cpu().numpy()
            else:
                image = image.numpy()
            image = image.transpose(1, 2, 0)
            # TODO: consider denormalizing image to ensure correct plotting (see extract_crf step of pipeline)
            # print(f"EvalDataset: Image after numpy: {image.shape}")
            ground_truth = np.array(Image.open(gt_path).convert('L'))
            prediction = np.array(Image.open(pred_path).convert('L'))
        else:
            image = np.array(Image.open(img_path).convert("RGB"))
            ground_truth = np.array(Image.open(gt_path).convert('L'))
            prediction = np.array(Image.open(pred_path).convert('L'))

        # Resize masks of image size not matching
        prediction = self._resize_mask(prediction, image)
        ground_truth = self._resize_mask(ground_truth, image)

        metadata = {'id': Path(img_path).stem, 'path': img_path, 'shape': tuple(image.shape[:2])}

        return (image, ground_truth, prediction, metadata)
    
    def _resize_mask(self, mask, image):
        # Check if sizes correspond
        H_im, W_im = image.shape[:2]
        H_mask, W_mask = mask.shape

        if (H_mask!= H_im or W_mask!=W_im):
            mask = cv2.resize(mask, dsize=(W_im, H_im), interpolation=cv2.INTER_NEAREST)  # (W, H) for cv2
        return mask

# if __name__ == '__main__':
#     root_dir = "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/US_MIXED/val"
#     gt_dir = "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/US_MIXED/val/lables"
#     pred_dir = "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/US_MIXED/val/predictions/maskcut_init_lr0.001_us_mixed_val_thresh0.0"

#     d = EvalDataset(root_dir,gt_dir,pred_dir)

            