#!/bin/bash

# Define parameter lists and their values
num_segments_values=(15)
image_ssd_beta_values=(0.0 1.0 2.0)
# 1.0 2.0

# Loop through each parameter combination
for num_segments in "${num_segments_values[@]}"; do
    for image_ssd_beta in "${image_ssd_beta_values[@]}"; do
        custom_path="mutinfo_train_carotid/diff_cluster_n/clusters${num_segments}_dino_ssd${image_ssd_beta}"
            sbatch dsp_single_pipeline.sh "$num_segments" "$image_ssd_beta" "$custom_path"
            # nohup bash dsp_single_pipeline.sh "$num_segments" "$image_ssd_beta" "$custom_path" > dsp_pipeline_"$image_ssd_beta".txt & 
    done
done