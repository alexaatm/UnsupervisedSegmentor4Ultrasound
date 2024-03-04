"""
Utility functions for evaluating the segmentation.
Adapted and largley modified version of source: https://github.com/lukemelas/deep-spectral-segmentation . All credits from the source code go to their respective authors.
"""

import numpy as np
from joblib import Parallel
from joblib.parallel import delayed
from scipy.optimize import linear_sum_assignment


def hungarian_match(flat_preds, flat_targets, preds_k, targets_k, metric='acc', n_jobs=16, thresh=0):
    assert (preds_k == targets_k)  # one to one
    num_k = preds_k

    # perform hungarian matching
    # print('Using iou as metric')
    if thresh != 0:
        results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(get_iou_with_thresh)(
            flat_preds, flat_targets, c1, c2, thresh) for c2 in range(num_k) for c1 in range(num_k))
    else:
        results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(get_iou)(
            flat_preds, flat_targets, c1, c2) for c2 in range(num_k) for c1 in range(num_k))
    results = np.array(results)
    results = results.reshape((num_k, num_k)).T
    match = linear_sum_assignment(flat_targets.shape[0] - results)
    match = np.array(list(zip(*match)))
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    # return a list of tuples (matches classes) and a full IoU matrix
    return res, results


def majority_vote(flat_preds, flat_targets, preds_k, targets_k, n_jobs=16, thresh=0):
    """
    NOTE: this function finds a GT (ground truth) class for every PL (pseudo label)
    based on the maximum IoU value.
    '
    iou_mat_reshaped = iou_mat.reshape((targets_k, preds_k)).T
    results = np.argmax(iou_mat_reshaped, axis=1)
    '
    Note transposed iou_mat. 
    Rows are PL, columns are GT. 
    So a single GT class can be assigned to multiple PL classes.
    """
    if thresh != 0 :
        # use threshold
        iou_mat = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(get_iou_with_thresh)(
            flat_preds, flat_targets, c1, c2, thresh) for c2 in range(targets_k) for c1 in range(preds_k))
    else:
        # dont use threshold
        iou_mat = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(get_iou)(
            flat_preds, flat_targets, c1, c2) for c2 in range(targets_k) for c1 in range(preds_k))
    iou_mat = np.array(iou_mat)
    iou_mat_reshaped = iou_mat.reshape((targets_k, preds_k)).T
    # TODO: visualize confusion matrix
    results = np.argmax(iou_mat_reshaped, axis=1)
    # match = np.array(list(zip(range(preds_k), results)))
    match = list(zip(range(preds_k), results))
    # return a list of tuples (matches classes) and a full IoU matrix
    return match, iou_mat_reshaped

def majority_vote_unique(flat_preds, flat_targets, preds_k, targets_k, n_jobs=16, thresh=0):
    """
    NOTE: this function finds a PL (pseudo label) class for every GT (ground truth)
    based on the maximum IoU value.
    '
    iou_mat_reshaped = iou_mat.reshape((targets_k, preds_k))
    results = np.argmax(iou_mat_reshaped, axis=1)
    '
    Note the ABSENCE of the transposed iou_mat.
    Rows are GT, columns are PL. 
    So a single GT class gets assigned only a single PL class.
    """
    if thresh != 0 :
        # use threshold
        iou_mat = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(get_iou_with_thresh)(
            flat_preds, flat_targets, c1, c2, thresh) for c2 in range(targets_k) for c1 in range(preds_k))
    else:
        # print('No threshold used')
        # dont use threshold
        iou_mat = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(get_iou)(
            flat_preds, flat_targets, c1, c2) for c2 in range(targets_k) for c1 in range(preds_k))
    iou_mat = np.array(iou_mat)
    iou_mat_reshaped = iou_mat.reshape((targets_k, preds_k))
    results = np.argmax(iou_mat_reshaped, axis=1)
    match = list(zip(results, range(preds_k)))
    # transpose the iou_matrix to match the format of hungarian matching output
    iou_mat_reshaped = iou_mat_reshaped.T
    # return a list of tuples (matches classes) and a full IoU matrix
    return match, iou_mat_reshaped

def get_iou(flat_preds, flat_targets, c1, c2):
    tp = 0
    fn = 0
    fp = 0
    tmp_all_gt = (flat_preds == c1)
    tmp_pred = (flat_targets == c2)
    tp += np.sum(tmp_all_gt & tmp_pred)
    fp += np.sum(~tmp_all_gt & tmp_pred)
    fn += np.sum(tmp_all_gt & ~tmp_pred)
    jac = float(tp) / max(float(tp + fp + fn), 1e-8)
    return jac

def get_iou_with_thresh(flat_preds, flat_targets, c1, c2, thresh = 0.5):
    tp = 0
    fn = 0
    fp = 0
    tmp_all_gt = (flat_preds == c1)
    tmp_pred = (flat_targets == c2)
    tp += np.sum(tmp_all_gt & tmp_pred)
    fp += np.sum(~tmp_all_gt & tmp_pred)
    fn += np.sum(tmp_all_gt & ~tmp_pred)
    jac = float(tp) / max(float(tp + fp + fn), 1e-8)
    if jac < thresh:
        jac = 0.0
    return jac

def make_labels_consecutive(labelmap):
    """
    NOTE: this function remaps each labelmap to have consecutive labels: 0, 1,2,3,4...K
    in case the labels present in the image had some labels missing:

    eg. INPUT labelmap has unique labels: [0, 1, 6, 9], then
    the OUTPUT labelmap would be: [0, 1, 2, 3]. and the matching would be:
    [(0,0), (1,1), (6, 2), (9,3)]

    Return: new labelmap, and the matching list. 
    """

    input_unique_labels = np.unique(labelmap)
    output_consecutive_labels = np.array(range(len(input_unique_labels)))
    matching = list(zip(input_unique_labels, output_consecutive_labels))
    new_labelmap = remap_labels(labelmap, matching)
    return new_labelmap, matching


def remap_labels(labelmap, matching):
    """
    NOTE: Remap each labelmap according to a new matching.

    Return: new labelmap, and the matching list. 
    """
    # unique_labels = np.unique(labelmap)
    new_labelmap = np.zeros_like(labelmap)
    for old_label_i, target_label_i in matching:
        # assert(old_label_i in unique_labels)
        new_labelmap[labelmap == int(old_label_i)] = int(target_label_i)
    return new_labelmap

def match(pred, gt, thresh):
    """
    Match prediction labels to ground truth labels
    based on the highest IoU value.

    Return: match between pred and ground truth, iou matrix. 
    """
    gt_unique = np.unique(gt)
    pred_unique = np.unique(pred)
    # print(f'GT unique labels: {gt_unique}')
    # print(f'PRED unique labels: {pred_unique}')
    n_clusters = len(pred_unique)
    n_classes = len(gt_unique)

    if (n_clusters == n_classes):
        # print('n_clusters == n_classes: Using hungarian algorithm for matching')
        match, iou_mat  = hungarian_match(pred, gt, preds_k=n_clusters, targets_k=n_classes, metric='iou', thresh=thresh)
    else:
        # print('n_clusters != n_classes: Using majority vote algorithm for matching')
        match, iou_mat = majority_vote_exclusive(pred, gt, preds_k=n_clusters, targets_k=n_classes, thresh=thresh)
    
    # print(f'Optimal matching: {match}')

    return match, iou_mat

def remap_match(match, gt_mapping, pred_mapping):
    """
    Map the given match to correspond to the original
    labels of gt and pred, based on gt_mapping and 
    pred_mapping

    Return: corrected mapp correpsonding to original values of labels. 
    """
    corrected_match = []
    for pred_i, gt_i in match:
        
        corrected_pred_i = -1
        for pred_original_j, pred_mapped_j in pred_mapping:
            if pred_mapped_j == pred_i:
                corrected_pred_i = pred_original_j                

        for gt_original_k,  gt_mapped_k in gt_mapping:
            if gt_mapped_k == gt_i:
                corrected_gt_i = gt_original_k
        
        corrected_match.append((corrected_pred_i, corrected_gt_i))
    
    return corrected_match
    
def majority_vote_exclusive(flat_preds, flat_targets, preds_k, targets_k, n_jobs=16, thresh=0):
    """
    This function finds a GT (ground truth) class for every PL (pseudo label)
    based on the maximum IoU value, but ensures that a GT class is assigned
    to a single PL class exclusively.
    """
    # Initialize an empty array to store the matching results
    match_gt2pr = {target_i: -1 for target_i in range(targets_k)}
    match_pr2gt = {pred_i: -1 for pred_i in range(preds_k)}


    # Calculate IoU 
    iou_mat = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(get_iou)(
         flat_preds, flat_targets, c1, c2) for c2 in range(targets_k) for c1 in range(preds_k))
    iou_mat = np.array(iou_mat)

    iou_mat = iou_mat.reshape((targets_k, preds_k))
    results_gt2pr = np.argmax(iou_mat, axis=1)
    iou_mat2 = iou_mat.reshape((targets_k, preds_k)).T
    results_pr2gt = np.argmax(iou_mat2, axis=1)

    # print('iou results_gt2pr: ', results_gt2pr)
    # print('iou results_pr2gt: ', results_pr2gt)

    # print('iou 1: ', iou_mat)
    # print('iou 2: ', iou_mat2)

    for gt_i in range(targets_k):
        if match_gt2pr[gt_i]==-1: # Check if the GT class has no matching PR yet
            match_gt2pr[gt_i]=results_gt2pr[gt_i]
            if  match_pr2gt[results_gt2pr[gt_i]]==-1: # Check if PR is not assigned to a GT yet
                match_pr2gt[results_gt2pr[gt_i]]=gt_i
            else:# Check which PR has higher IoU
                current_pr_i = results_gt2pr[gt_i]
                last_assigned_gt_i = match_pr2gt[current_pr_i]
                candidate_gt_i = gt_i
                if (iou_mat[candidate_gt_i][current_pr_i] > iou_mat[last_assigned_gt_i][current_pr_i]):
                    match_pr2gt[results_gt2pr[gt_i]]=candidate_gt_i
                    match_gt2pr[last_assigned_gt_i]=-1
                else:
                    match_pr2gt[results_gt2pr[gt_i]]=last_assigned_gt_i
                    match_gt2pr[candidate_gt_i]=-1
        
    # print("match_gt2pr: ", match_gt2pr)
    # print("match_pr2gt: ", match_pr2gt)

    match = [(match_gt2pr[gt_i], gt_i) for gt_i in match_gt2pr]
    return match, iou_mat2

from collections import defaultdict

def process_matches(matches):
    """
    This function looks at all the matches made per image, and 
    returns a single - final- mapping, where the majority vote 
    pseudo label is chosen as a final match. Additionally, consistensy
    of the matchings over the dataset is presented in %. 100% means
    a complete agreement in matching for all images.
    """
    final_mapping = {}  # Dictionary to store the final mapping
    final_mapping_with_unmatched = {}  # Dictionary to store the mapping including unmatched cases
    gt_counts = defaultdict(lambda: defaultdict(int))  # Dictionary to store counts of GT-PL pairs
    mapping_consistency = {}

    # Iterate over each image's matches
    for match_list in matches:
        # print(f"match_list={match_list}")
        for pl, gt in match_list:
            gt_counts[gt][pl] += 1

    # Find the most frequent PL for each GT
    for gt, pl_counts in gt_counts.items():
        if pl_counts:
            max_pl = max(pl_counts, key=pl_counts.get)
            final_mapping_with_unmatched[gt]=max_pl

            # If the maximum count corresponds to -1, choose the second winner if available
            if max_pl == -1:
                sorted_counts = sorted(pl_counts.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_counts) > 1:
                    final_mapping[gt] = sorted_counts[1][0]
                else:
                    final_mapping[gt] = -1
            else:
                final_mapping[gt] = max_pl

    # Print final mapping
    print("Final Mapping:")
    for gt, pl in final_mapping.items():
        print(f"GT {gt} -> PL {pl}")

    print("Final Mapping including Unmatched GT:")
    for gt, pl in final_mapping_with_unmatched.items():
        print(f"GT {gt} -> PL {pl}")

    # Print percentage of consistent mappings for each ground truth class
    print("\nPercentage of Consistent Mappings:")
    for gt, pl_counts in gt_counts.items():
        if pl_counts:
            total_counts = sum(pl_counts.values())
            percentage = (pl_counts[final_mapping[gt]] / total_counts) * 100
            mapping_consistency[gt]=percentage
            print(f"GT {gt}: {percentage:.2f}% consistent mappings (PL {final_mapping[gt]})")

    # Return lists of PL labels (in order of GT labels)
    fin_map = [pl for gt, pl in final_mapping.items()]
    fin_map_with_unmatched = [pl for gt, pl in final_mapping_with_unmatched.items()]
    consistency_map = [percentage for gt, percentage in mapping_consistency.items()]
    avg_consistency = sum(consistency_map) / len(consistency_map)
    return (fin_map, fin_map_with_unmatched, consistency_map), avg_consistency


import numpy as np
import cv2

def boundary_recall_with_distance(gt_boundary, pred_boundary, d=0):
    """
    Calculate Boundary Recall
    Ref. for the formula:
    https://www.tu-chemnitz.de/etit/proaut/publications/neubert_protzel_superpixel.pdf
    """

    # Dilate the boundary maps to include neighboring pixels within distance d
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*d+1, 2*d+1))
    dilated_pred_boundary = cv2.dilate(pred_boundary.astype(np.uint8), kernel)
    
    # Calculate true positives, false positives, false negatives
    tp = np.sum(gt_boundary & dilated_pred_boundary)
    fp = np.sum(~gt_boundary & dilated_pred_boundary)
    fn = np.sum(gt_boundary & ~dilated_pred_boundary)
    
    # Compute boundary recall
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    
    # return (recall, gt_boundary, dilated_pred_boundary)
    return recall


import numpy as np
from skimage import measure

# Function to create boundary image from binary mask using scikit-image
def create_boundary_image_full(labelmap):
    boundary_image = np.zeros_like(labelmap, dtype=np.uint8)

    for i in np.unique(labelmap):
        # find all contours of segments with label i
        binary_image = (labelmap == i)
        contours = measure.find_contours(binary_image, 0.5)

        # Draw contours on the boundary image
        for contour in contours:
            contour = np.round(contour).astype(int)  # Convert to integer coordinates
            for point in contour:
                boundary_image[point[0], point[1]] = 1
    
    return boundary_image
