"""
Evaluation functions for assessing the segmentation.
Adapted from: https://github.com/lukemelas/deep-spectral-segmentation

What is changed is enabling to perform this evaluataion on any given 
labeled dataset. See evaluation/dataset.py  - a dataset class that stores 
images, predictions and ground truth. Additionally, wandb logging of 
segmentations, metrics in tables and IoU confusion matrices is done.
"""

import os
from pathlib import Path
import numpy as np
import wandb
from matplotlib.cm import get_cmap
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from skimage.color import label2rgb
from tqdm import tqdm, trange

# evaluation utilities
import eval_utils

# for reading a dataset with groundth truth and labels
from dataset import EvalDataset

# logging
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import datetime
import pandas as pd
import random

def evaluate_dataset(dataset_dir, n_classes, n_clusters):
    dataset = EvalDataset(dataset_dir)
    
    # Add background class
    # n_classes = cfg.n_classes + 1 #TODO: check if you need this
    if n_clusters is None:
        n_clusters = n_classes

    # Iterate
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes

    # Load all pixel embeddings
    # TODO: change to size from the dataset
    all_preds = np.zeros((len(dataset) * 500 * 500), dtype=np.float32)
    all_gt = np.zeros((len(dataset) * 500 * 500), dtype=np.float32)
    offset_ = 0

    for i in trange(len(dataset), desc='Concatenating all predictions'):
        image, target, mask, metadata = dataset[i]
        # Check where ground-truth is valid and append valid pixels to the array
        valid = (target != 255)
        n_valid = np.sum(valid)
        all_gt[offset_:offset_+n_valid] = target[valid]
        # Append the predicted targets in the array
        all_preds[offset_:offset_+n_valid, ] = mask[valid]
        all_gt[offset_:offset_+n_valid, ] = target[valid]
        # Update offset_
        offset_ += n_valid

    # Truncate to the actual number of pixels
    all_preds = all_preds[:offset_, ]
    all_gt = all_gt[:offset_, ]

    # Do hungarian matching 
    if n_clusters == n_classes:
        print('Using hungarian algorithm for matching')
        match, iou_mat  = eval_utils.hungarian_match(all_preds, all_gt, preds_k=n_clusters, targets_k=n_classes, metric='iou')
    else:
        print('Using majority voting for matching')
        match, iou_mat = eval_utils.majority_vote(all_preds, all_gt, preds_k=n_clusters, targets_k=n_classes)
    print(f'Optimal matching: {match}')

    # Remap predictions
    num_elems = offset_
    reordered_preds = np.zeros(num_elems, dtype=all_preds.dtype)
    for pred_i, target_i in match:
        reordered_preds[all_preds == int(pred_i)] = int(target_i)

    # TP, FP, and FN evaluation
    for i_part in range(0, n_classes):
        tmp_all_gt = (all_gt == i_part)
        tmp_pred = (reordered_preds == i_part)
        tp[i_part] += np.sum(tmp_all_gt & tmp_pred)
        fp[i_part] += np.sum(~tmp_all_gt & tmp_pred)
        fn[i_part] += np.sum(tmp_all_gt & ~tmp_pred)

    # Calculate Jaccard index
    jac = [0] * n_classes
    for i_part in range(0, n_classes):
        jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

    # Print results
    eval_result = dict()
    eval_result['jaccards_all_categs'] = jac
    eval_result['mIoU'] = np.mean(jac)
    eval_result['IoU_matrix'] = iou_mat
    print('Evaluation of semantic segmentation ')
    
    return eval_result, match

def visualize(dataset_dir, n_classes, vis_dir: str = './vis', save_rand_k: int = 1):
    # Set up dataset
    dataset = EvalDataset(dataset_dir)

    # Images for wandb logging
    img_list = []
    img_list2 = []

    # Visualize
    if save_rand_k > len(dataset): # a safeguard for a case when more samples are asked than is in dataset
        random_inds = [0]
    elif save_rand_k > 0:
        random_inds = random.sample(range(len(dataset)), save_rand_k)
    else:
        random_inds = -1 #no images will be sampled

    print("images to sample in wandb: ", random_inds)
    
    vis_dir = Path(vis_dir)
    colors = get_cmap('tab20', n_classes + 1).colors[:,:3]
    pbar = tqdm(dataset, total=len(dataset), desc='Saving visualizations: ')
    for i, (image, target, mask, metadata) in enumerate(pbar):
        image = np.array(image)
        target = np.array(target)
        target[target == 255] = 0  # set the "unknown" regions to background for visualization
        # Overlay mask on image
        image_pred_overlay = label2rgb(label=mask, image=image, colors=colors[np.unique(target)[1:]], bg_label=0, alpha=0.45)
        image_target_overlay = label2rgb(label=target, image=image, colors=colors[np.unique(target)[1:]], bg_label=0, alpha=0.45)
        # Save
        image_id = metadata["id"]
        path_pred = vis_dir / 'pred' / f'{image_id}-pred.png'
        path_target = vis_dir / 'target' / f'{image_id}-target.png'
        path_pred.parent.mkdir(exist_ok=True, parents=True)
        path_target.parent.mkdir(exist_ok=True, parents=True)

        pred_img = Image.fromarray((image_pred_overlay * 255).astype(np.uint8))
        pred_img.save(str(path_pred))
        target_img = Image.fromarray((image_target_overlay * 255).astype(np.uint8))
        target_img.save(str(path_target))
        if i in random_inds:
            img_list.append([image_id, image, pred_img, target_img])
            img_list2.append([image_id, image, mask, target])

    print(f'Saved visualizations to {vis_dir.absolute()}')
    return img_list, img_list2


@hydra.main(version_base=None, config_path="./configs", config_name="defaults")
def main(cfg: DictConfig):
    # Logging
    if cfg.wandb:
        wandb.login(key=cfg.wandb.key)
        cfg.wandb.key=""
        wandb.init(name ="eval_" + cfg.dataset.name + "_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), project=cfg.wandb.setup.project, config=OmegaConf.to_container(cfg), save_code=True)
        cfg = DictConfig(wandb.config.as_dict())  # get the config back from wandb for hyperparameter sweeps

    # Configuration
    print(OmegaConf.to_yaml(cfg))
    print(f'Current working directory: {os.getcwd()}')

    # Evaluate
    eval_stats, match = evaluate_dataset(cfg.dataset.dataset_dir, cfg.dataset.n_classes, cfg.dataset.get('n_clusters', None))
    print(eval_stats)
    print(match)

    # Visualize
    img_list, img_list2 = visualize(cfg.dataset.dataset_dir, cfg.dataset.n_classes, cfg.vis_dir, 1)

    if cfg.wandb:
        wandb.log({'mIoU': eval_stats['mIoU']})
        wandb.log({'jac': eval_stats['jaccards_all_categs']}) 
        wandb.log({'match': match})
        wandb.log({'IoU_matrix': eval_stats['IoU_matrix']})

        # Log confusion matrix and other metrics to wandb
        class_names = [f'class{i}' for i in range(cfg.dataset.n_classes)]
        iou_df = pd.DataFrame(data=eval_stats['IoU_matrix'], index=class_names, columns=class_names)
        wandb.log({'IoU_heatmap': wandb.plots.HeatMap(class_names, class_names, iou_df, show_text=True)})

        # Log Jaccard index table
        wandb.log({"jaccard_table": wandb.Table(data=[eval_stats['jaccards_all_categs']], columns=class_names)})

        # Log segment matchings
        pred_gt_list = [[pr, gt] for pr, gt in match]
        wandb.log({"match_table": wandb.Table(data=pred_gt_list, columns=['Pseudo label','Ground Truth label'])})

        # Log example images
        img_table = wandb.Table(columns=['ID', 'Image', 'Pred', 'Ground_Truth'])
        for img_id, img, pred, gt in img_list:
            img_table.add_data(img_id, wandb.Image(img), wandb.Image(pred), wandb.Image(gt))
        wandb.log({"Example Images" : img_table})

        # Visualize labels using wandb
        img_table2 = wandb.Table(columns=['ID', 'Image'])
        for img_id, img, pred, gt in img_list2:
            mask_img = wandb.Image(img, masks = {
                "prediction" : {"mask_data" : pred},
                "groud_truth" : {"mask_data" : gt},
            })
            img_table2.add_data(img_id, mask_img)
        wandb.log({"Example Images2" : img_table2})


if __name__ == '__main__':
    main()
