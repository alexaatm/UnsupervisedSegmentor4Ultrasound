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
import cv2

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

def evaluate_dataset(dataset, n_classes, n_clusters, thresh):

    if dataset.n_clusters is not None:
        n_clusters = dataset.n_clusters
    elif n_clusters is None:
        n_clusters = n_classes
    
    # Iterate
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes

    matches = []
    iou_matrices = []
    remapped_preds = []

    for i in trange(len(dataset), desc='Iterating predictions'):
        image, gt, pred, metadata = dataset[i]

        # Do matching 
        gt_unique = np.unique(gt)
        pred_unique = np.unique(pred)
        print(f'GT unique labels: {gt_unique}')
        print(f'PRED unique labels: {pred_unique}')
        if np.array_equal(gt_unique,pred_unique) and n_clusters==n_classes:
        # if len(gt_unique)==len(pred_unique):
            print('Using hungarian algorithm for matching')
            match, iou_mat  = eval_utils.hungarian_match(pred, gt, preds_k=n_clusters, targets_k=n_classes, metric='iou', thresh=thresh)
        else:
            print('Using majority voting for matching')
            match, iou_mat = eval_utils.majority_vote_unique(pred, gt, preds_k=n_clusters, targets_k=n_classes, thresh=thresh)
        print(f'Optimal matching: {match}')
        matches.append(match)
        iou_matrices.append(iou_mat)

        # reorder prediction according to found mapping
        reordered_pred = np.zeros_like(pred)
        for pred_i, target_i in match:
            reordered_pred[pred == int(pred_i)] = int(target_i)
        remapped_preds.append(reordered_pred)
        
        # TP, FP, and FN evaluation
        for i_part in range(0, n_classes):
            tmp_gt = (gt == i_part) #get class i mask from ground truth
            tmp_pred = (reordered_pred == i_part) #get class i mask from predictions
            tp[i_part] += np.sum(tmp_gt & tmp_pred)
            fp[i_part] += np.sum(~tmp_gt & tmp_pred)
            fn[i_part] += np.sum(tmp_gt & ~tmp_pred)

    # Calculate Jaccard index
    jac = [0] * n_classes
    for i_part in range(0, n_classes):
        jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

    # Print results
    eval_result = dict()
    eval_result['jaccards_all_categs'] = jac
    eval_result['mIoU'] = np.mean(jac)
    eval_result['IoU_matrix'] = iou_matrices
    print('Evaluation of semantic segmentation ')
    
    return eval_result, matches, remapped_preds

def evaluate_dataset_with_single_matching(dataset, n_classes, n_clusters, thresh):
    # dataset = EvalDataset(dataset_dir)
    
    # Add background class
    # n_classes = cfg.n_classes + 1 #TODO: check if you need this
    if dataset.n_clusters is not None:
        n_clusters = dataset.n_clusters
    elif n_clusters is None:
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
        match, iou_mat  = eval_utils.hungarian_match(all_preds, all_gt, preds_k=n_clusters, targets_k=n_classes, metric='iou', thresh=thresh)
    else:
        print('Using majority voting for matching')
        match, iou_mat = eval_utils.majority_vote_unique(all_preds, all_gt, preds_k=n_clusters, targets_k=n_classes, thresh=thresh)
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

def visualize(dataset, inds_to_vis, vis_dir: str = './vis'):
    # Images for wandb logging
    img_list = []

    vis_dir = Path(vis_dir)

    # Get colors (using 21 as number of classes by default)
    # TODO: what to do when need more colors https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib
    colors1 = get_cmap('tab20', 21).colors[:,:3]
    colors2 = get_cmap('tab20b', 21).colors[:,:3]
    colors = np.concatenate((colors1, colors2), axis=0)


    # Create a legend image
    legend_image = np.zeros((len(colors) * 20, 100, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        legend_image[i * 20: (i + 1) * 20, :] = color * 255
        cv2.putText(legend_image, str(i), (5, i * 20 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save the legend image
    legend_path = vis_dir / 'legend.png'
    legend_path.parent.mkdir(exist_ok=True, parents=True)
    legend_img = Image.fromarray(legend_image)
    legend_img.save(str(legend_path))


    pbar = tqdm(dataset, total=len(dataset), desc='Saving visualizations: ')
    for i, (image, target, mask, metadata) in enumerate(pbar):
        if i in inds_to_vis:
            image = np.array(image)
            target = np.array(target)
            mask = np.array(mask)
            target[target == 255] = 0  # set the "unknown" regions to background for visualization
            

            # Check if sizes correspond
            H_im, W_im = image.shape[:2]
            H_gt, W_gt = target.shape
            H_pr, W_pr = mask.shape

            H = np.max([H_im, H_gt, H_pr])
            W = np.max([W_im, W_gt, W_pr])

            # print(f'Image shape: {image.shape}')
            # print(f'Gt shape: {target.shape}')
            # print(f'Pred shape: {mask.shape}')


            if (H_gt!= H or W_gt!=W):
                print("GT needs to be resized")
                gt_im_res = cv2.resize(target, dsize=(W, H), interpolation=cv2.INTER_NEAREST)  # (H, W)
                gt_im_res[:target.shape[0], :target.shape[1]] = target  # replace with the initial groundtruth version, just in case they are different
                target = gt_im_res

            if (H_pr!= H or W_pr!=W):
                print("PRED needs to be resized")
                pred_im_res = cv2.resize(mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)  # (H, W)
                pred_im_res[:mask.shape[0], :mask.shape[1]] = mask  # replace with the initial prediction version, just in case they are different
                mask = pred_im_res

            # print(f'After:')
            # print(f'Image shape: {image.shape}')
            # print(f'Gt shape: {target.shape}')
            # print(f'Pred shape: {mask.shape}')
            
            # Overlay mask on image
            image_pred_overlay = label2rgb(label=mask, image=image, colors=colors[np.unique(mask)[1:]], bg_label=0, alpha=0.45)
            image_target_overlay = label2rgb(label=target, image=image, colors=colors[np.unique(target)[1:]], bg_label=0, alpha=0.45)
            
            # Save samples
            image_id = metadata["id"]
            path_pred = vis_dir / 'pred' / f'{image_id}-pred.png'
            path_target = vis_dir / 'target' / f'{image_id}-target.png'
            path_pred.parent.mkdir(exist_ok=True, parents=True)
            path_target.parent.mkdir(exist_ok=True, parents=True)

            pred_img = Image.fromarray((image_pred_overlay * 255).astype(np.uint8))
            pred_img.save(str(path_pred))
            target_img = Image.fromarray((image_target_overlay * 255).astype(np.uint8))
            target_img.save(str(path_target))
            img_list.append([image_id, image, pred_img, target_img])


    print(f'Saved visualizations to {vis_dir.absolute()}')
    return img_list, legend_img


@hydra.main(version_base=None, config_path="./configs", config_name="defaults")
def main(cfg: DictConfig):
    if cfg.eval_per_image:
        # Evaluate per image

        # Logging
        if cfg.wandb:
            wandb.login(key=cfg.wandb.key)
            cfg.wandb.key=""
            wandb.init(name ="eval_" + cfg.dataset.name + "_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), project=cfg.wandb.setup.project, config=OmegaConf.to_container(cfg), save_code=True,
                       tags=['fixed',cfg.wandb.tag])
            cfg = DictConfig(wandb.config.as_dict())  # get the config back from wandb for hyperparameter sweeps

        # Configuration
        print(OmegaConf.to_yaml(cfg))
        print(f'Current working directory: {os.getcwd()}')

        # Evaluate
        dataset = EvalDataset(cfg.dataset.dataset_dir, cfg.dataset.gt_dir, cfg.dataset.pred_dir)
        eval_stats, matches, preds = evaluate_dataset(dataset, cfg.dataset.n_classes, cfg.dataset.get('n_clusters', None), cfg.iou_thresh)
        print("eval stats:", eval_stats)
        print("matches:", matches)


        # Visualize some image evaluation samples
        random.seed(1) 
        if cfg.vis_rand_k > len(dataset): # a safeguard for a case when more samples are asked than is in dataset
            inds_to_vis = [0]
        elif cfg.vis_rand_k > 0:
            inds_to_vis = random.sample(range(len(dataset)), cfg.vis_rand_k)
        else:
            inds_to_vis = [-1] #no images will be sampled

        print("sample images to visualize in wandb: ", inds_to_vis)

        # Visualize
        img_list, legend = visualize(dataset, inds_to_vis, cfg.vis_dir)

        if cfg.wandb:
            wandb.log({'mIoU': eval_stats['mIoU']})

            # Log metrics and sample evaluation results
            class_names_all = [f'GT_class{i}' for i in range(cfg.dataset.n_classes)]
            if dataset.n_clusters is not None:
                n_clusters = dataset.n_clusters
            elif cfg.dataset.n_clusters is not None:
                n_clusters = cfg.dataset.n_clusters
            else:
                n_clusters = n_classes
            pseudolabel_names = [f'PL_class{i}' for i in range(n_clusters)]
            
            # Table for logging segment matching
            match_table = wandb.Table(columns = ['ID'] + class_names_all)
    
            # Table for logging corrected labels using wandb
            remapped_pred_table = wandb.Table(columns=['ID', 'Image'])

            # Go through dataset and log data
            for i, (sample, iou_m, match, remapped_pred) in enumerate(zip(dataset, eval_stats['IoU_matrix'], matches, preds)):
                im, target, pred, metadata = sample
                id = metadata['id']

                # log matches for every sample in dataset - in a table
                row_data = [id] + [None] * cfg.dataset.n_classes
                for pr, gt in match:
                    row_data[1 + gt] = pr if row_data[1 + gt] is None else row_data[1 + gt]
                match_table.add_data(*row_data)

                # log IoU heatmaps and remapped preds only for selected samples
                if i in inds_to_vis:
                    # get lists of unique labels to name the columns of heatmaps (may be different for each image)
                    # class_names = [f'GT_class{i}' for i in np.unique(target)]
                    # pseudolabel_names = [f'PL_class{i}' for i in np.unique(pred)]

                    # log IoU heatmaps - individually (cannot log wandb heatmaps in a wandb table...)
                    iou_df = pd.DataFrame(data=iou_m, index=pseudolabel_names, columns=class_names_all)
                    heatmap = wandb.plots.HeatMap(class_names_all,pseudolabel_names, iou_df, show_text=True)
                    wandb.log({f"Sample IoU Heatmap {id}": heatmap})

                    # log remapped predictions - in a table
                    mask_img = wandb.Image(im, masks = {
                    "groud_truth" : {"mask_data" : target},
                    "prediction" : {"mask_data" : pred},
                    "remapped_pred" : {"mask_data" : remapped_pred},
                    })
                    remapped_pred_table.add_data(id, mask_img)

            # Log completed tables
            wandb.log({"match_table": match_table})
            wandb.log({"Example Images After Remapping" : remapped_pred_table})

            # Log Jaccard index table
            wandb.log({"jaccard_table": wandb.Table(data=[eval_stats['jaccards_all_categs']], columns=class_names_all)})

            # Log example images with overlayed segmentation
            img_table = wandb.Table(columns=['ID', 'Image', 'Pred', 'Ground_Truth'])
            for img_id, img, pred, gt in img_list:
                img_table.add_data(img_id, wandb.Image(img), wandb.Image(pred), wandb.Image(gt))
            wandb.log({"Example Images" : img_table})

            # Log legend to interprete example images
            wandb.log({"Legend CLass-Color": wandb.Image(legend)})

            wandb.finish()

    else:
        # Evaluate over the whole dataset

        # Logging
        if cfg.wandb:
            wandb.login(key=cfg.wandb.key)
            cfg.wandb.key=""
            wandb.init(name ="eval_" + cfg.dataset.name + "_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), project=cfg.wandb.setup.project, config=OmegaConf.to_container(cfg), save_code=True,
                tags=['fixed', cfg.wandb.tag, 'majority_vote_unique'])
            cfg = DictConfig(wandb.config.as_dict())  # get the config back from wandb for hyperparameter sweeps

        # Configuration
        print(OmegaConf.to_yaml(cfg))
        print(f'Current working directory: {os.getcwd()}')

        # Evaluate
        dataset = EvalDataset(cfg.dataset.dataset_dir, cfg.dataset.gt_dir, cfg.dataset.pred_dir)
        eval_stats, match = evaluate_dataset_with_single_matching(dataset, cfg.dataset.n_classes, cfg.dataset.get('n_clusters', None), cfg.iou_thresh)
        # print(eval_stats)
        # print(match)

        # Visualize some image evaluation samples
        random.seed(1)
        if cfg.vis_rand_k > len(dataset): # a safeguard for a case when more samples are asked than is in dataset
            inds_to_vis = [0]
        elif cfg.vis_rand_k > 0:
            inds_to_vis = random.sample(range(len(dataset)), cfg.vis_rand_k)
        else:
            inds_to_vis = [-1] #no images will be sampled

        print("sample images to visualize in wandb: ", inds_to_vis)
        img_list, legend = visualize(dataset, inds_to_vis, cfg.vis_dir)


        if cfg.wandb:
            wandb.log({'mIoU': eval_stats['mIoU']})

            # Log confusion matrix and other metrics to wandb
            class_names = [f'GT_class{i}' for i in range(cfg.dataset.n_classes)]
            if dataset.n_clusters is not None:
                n_clusters = dataset.n_clusters
            elif cfg.dataset.n_clusters is not None:
                n_clusters = cfg.dataset.n_clusters
            else:
                n_clusters = n_classes
            pseudolabel_names = [f'PL_class{i}' for i in range(n_clusters)]
            iou_df = pd.DataFrame(data=eval_stats['IoU_matrix'], index=pseudolabel_names, columns=class_names)
            wandb.log({'IoU_heatmap': wandb.plots.HeatMap(class_names, pseudolabel_names, iou_df, show_text=True)})

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

            # Log legend to interprete example images
            wandb.log({"Legend CLass-Color": wandb.Image(legend)})

            wandb.finish()

if __name__ == '__main__':
    main()
