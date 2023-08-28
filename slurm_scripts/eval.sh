#!/bin/bash

# Define parameter lists and their values
# pred_dirs=(
#     "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/deep-spectral-segmentation/outputs/pipeline/mutinfo_val_carotid/2023-08-14--13-59-23/semantic_segmentations/laplacian/crf_segmaps"
#     "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/deep-spectral-segmentation/outputs/pipeline/mutinfo_val_carotid/2023-08-13--20-47-28/semantic_segmentations/laplacian/crf_multi_region"
#     "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/deep-spectral-segmentation/outputs/pipeline/mutinfo_val_carotid/2023-08-16--18-13-13/semantic_segmentations/laplacian/crf_segmaps"
#     "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/deep-spectral-segmentation/outputs/pipeline/mutinfo_val_carotid/2023-08-16--18-13-13/semantic_segmentations/laplacian/crf_multi_region"
#     "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/deep-spectral-segmentation/outputs/pipeline/mutinfo_val_carotid/2023-08-16--22-45-22/semantic_segmentations/laplacian/crf_segmaps"
#     "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/deep-spectral-segmentation/outputs/pipeline/mutinfo_val_carotid/2023-08-16--22-45-22/semantic_segmentations/laplacian/crf_multi_region"
#   )

pred_dirs=(
    "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/deep-spectral-segmentation/outputs/mutinfo_val_carotid/diff_cluster_n/clusters6_dinossd1/semantic_segmentations/laplacian/crf_segmaps"
    "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/deep-spectral-segmentation/outputs/mutinfo_val_carotid/diff_cluster_n/clusters6_dinossd1/semantic_segmentations/laplacian/crf_multi_region"
    "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/deep-spectral-segmentation/outputs/mutinfo_val_carotid/diff_cluster_n/clusters6_dinossd2/semantic_segmentations/laplacian/crf_segmaps"
    "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/deep-spectral-segmentation/outputs/mutinfo_val_carotid/diff_cluster_n/clusters6_dinossd2/semantic_segmentations/laplacian/crf_multi_region"
    "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/deep-spectral-segmentation/outputs/mutinfo_val_carotid/diff_cluster_n/clusters6_onlydino/semantic_segmentations/laplacian/crf_segmaps"
    "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/deep-spectral-segmentation/outputs/mutinfo_val_carotid/diff_cluster_n/clusters6_onlydino/semantic_segmentations/laplacian/crf_multi_region"
    # "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/deep-spectral-segmentation/outputs/mutinfo_val_carotid/diff_cluster_n/clusters15_dinossd3/semantic_segmentations/laplacian/crf_segmaps"
    # "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/deep-spectral-segmentation/outputs/mutinfo_val_carotid/diff_cluster_n/clusters15_dinossd3/semantic_segmentations/laplacian/crf_multi_region"
  )

iou_thresh_values=(0.0)
eval_per_image_values=(False True)

# Loop through each parameter combination
for eval_per_image in "${eval_per_image_values[@]}"; do
    for iou_thresh in "${iou_thresh_values[@]}"; do
        for pred_dir in "${pred_dirs[@]}"; do
            sbatch single_eval.sh "$eval_per_image" "$iou_thresh" "$pred_dir"
        done
    done
done