"""
Utility functions for evaluating the segmentation.
Source: https://github.com/lukemelas/deep-spectral-segmentation
All credits for this code go to their respective authors
"""

import numpy as np
from joblib import Parallel
from joblib.parallel import delayed
from scipy.optimize import linear_sum_assignment


def hungarian_match(flat_preds, flat_targets, preds_k, targets_k, metric='acc', n_jobs=16, thresh=0):
    assert (preds_k == targets_k)  # one to one
    num_k = preds_k

    # perform hungarian matching
    print('Using iou as metric')
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
        print('No threshold used')
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
