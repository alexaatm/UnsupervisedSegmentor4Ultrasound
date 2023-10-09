#!/bin/bash

# Define parameter lists and their values
num_segments_values=(15)
image_ssd_beta_values=(0.0 1.0)
image_dino_gamma=(1.0)
image_var_values=(0.0 1.0 2.0)
# datasets=(liver_mixed_TEST thyroid_single_side_TEST thyroid_compounded_TEST)
# datasets=(carotid_mixed_TEST)
datasets=(carotid_mixed_VAL)
only_eval=False
# pipeline steps = (allTrue allFalse defaults, where defaults is all true, except eval)
pipeline_steps=allTrue
eval_per_image=True
eval_per_dataset=True


# Loop through each parameter combination
for dataset in "${datasets[@]}"; do
    for num_segments in "${num_segments_values[@]}"; do
        for image_ssd_beta in "${image_ssd_beta_values[@]}"; do
            for image_var in "${image_var_values[@]}"; do
                custom_path="${dataset}/diff_cluster_n/clusters${num_segments}_dino${image_dino_gamma}_ssd${image_ssd_beta}_var${image_var}"
                    sbatch dsp_single_pipeline.sh "$num_segments" "$image_ssd_beta" "$custom_path" "$image_dino_gamma" "$dataset" "$only_eval" "$pipeline_steps" "$eval_per_image" "$eval_per_dataset" "$image_var"
                    # bash dsp_single_pipeline.sh "$num_segments" "$image_ssd_beta" "$custom_path" > dsp_pipeline_"$image_ssd_beta".txt & 
            done
        done
    done
done