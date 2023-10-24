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
    def __init__(self, root_dir, gt_dir = "", pred_dir = "", check_size=True, transform: object = None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.gt_dir = gt_dir if gt_dir!="" else os.path.join(root_dir, 'ground_truth')
        self.pred_dir = pred_dir if pred_dir!="" else os.path.join(root_dir, 'predictions')
        self.image_list = os.listdir(self.image_dir)
        self.transform = transform

        print("root:", self.root_dir)
        print("image_dir:", self.image_dir)
        print("gt_dir:", self.gt_dir)
        print("pred_dir:", self.pred_dir)

        if check_size:
            print("Checking sizes of ground truth and predictions")
            self.check_sizes_and_resize()

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
            image = np.array(self.transform(Image.open(img_path).convert("RGB")))
            ground_truth = np.array(self.transform(Image.open(gt_path).convert('L')))
            prediction = np.array(self.transform(Image.open(pred_path).convert('L')))
        else:
            image = np.array(Image.open(img_path).convert("RGB"))
            ground_truth = np.array(Image.open(gt_path).convert('L'))
            prediction = np.array(Image.open(pred_path).convert('L'))

        metadata = {'id': Path(img_path).stem, 'path': img_path, 'shape': tuple(image.shape[:2])}

        return (image, ground_truth, prediction, metadata)
    
    def check_sizes_and_resize(self):
        segm_num = 0
        for img_name_full in tqdm(self.image_list):
            img_name = img_name_full[:-4]
            img_path = os.path.join(self.image_dir, img_name_full)
            gt_path = os.path.join(self.gt_dir, img_name + ".png")
            pred_path = os.path.join(self.pred_dir, img_name + ".png")

            image = np.array(Image.open(img_path).convert("RGB"))
            gt = np.array(Image.open(gt_path).convert('L'))
            pred = np.array(Image.open(pred_path).convert('L'))
            
            # get the number of unique labels to determine the number of segments per image
            unique_labels = np.unique(pred)
            print(f'unique labels per image: {unique_labels}')
            # current_segm_num = np.max(unique_labels)
            current_segm_num = len(unique_labels)
            # current_segm_num = max(len(unique_labels), np.max(unique_labels))
            print(f'current_segm_num: {current_segm_num}')
            if current_segm_num > segm_num:
                segm_num = current_segm_num

            # Check if sizes correspond
            H_im, W_im = image.shape[:2]
            H_gt, W_gt = gt.shape
            H_pr, W_pr = pred.shape

            # H = np.max([H_im, H_gt, H_pr])
            # W = np.max([W_im, W_gt, W_pr])
            H=H_im
            W=W_im

            if (H_gt!= H or W_gt!=W):
                gt_im_res = cv2.resize(gt, dsize=(W, H), interpolation=cv2.INTER_CUBIC)  # (H, W)
                # gt_im_res[:gt.shape[0], :gt.shape[1]] = gt  # replace with the initial groundtruth version, just in case they are different
                Image.fromarray(gt_im_res).convert('L').save(gt_path)
        
            if (H_pr!= H or W_pr!=W):
                pred_im_res = cv2.resize(pred, dsize=(W, H), interpolation=cv2.INTER_CUBIC)  # (H, W)
                # pred_im_res[:pred.shape[0], :pred.shape[1]] = pred  # replace with the initial prediction version, just in case they are different
                Image.fromarray(pred_im_res).convert('L').save(pred_path)
        
        self.n_clusters = segm_num
        print(f'self.n_clusters: {self.n_clusters}')
        self.H=H_im
        self.W=W_im


# if __name__ == '__main__':
#     root_dir = "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/US_MIXED/val"
#     gt_dir = "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/US_MIXED/val/lables"
#     pred_dir = "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/US_MIXED/val/predictions/maskcut_init_lr0.001_us_mixed_val_thresh0.0"

#     d = EvalDataset(root_dir,gt_dir,pred_dir)

            