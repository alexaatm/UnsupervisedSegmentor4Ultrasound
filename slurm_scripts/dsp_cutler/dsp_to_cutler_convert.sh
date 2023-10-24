#!/bin/sh
 
#SBATCH --job-name=dsp2Cutler
#SBATCH --output=dsp2Cutler-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=dsp2Cutler-%A.err  # Standard error of the script
#SBATCH --gres=gpu:0  # Number of GPUs if needed
#SBATCH --cpus-per-task=8  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=36G  # Memory in GB (Don't use more than 126G per GPU)
 
# activate corresponding environment
source ../../../.venv/bin/activate

# navigate to the cutler directory
cd ../../CutLER/cutler/tools

export DETECTRON2_DATASETS=/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data

dataset_dir=$1
pred_dir=$2
output_name=$3

# dsp_labelmaps_clusters15_dino_ssd0_crf_segmaps.json

echo dataset_dir="${dataset_dir}"
echo pred_dir="${pred_dir}"
echo output_name="${output_name}"

python labelmap_to_cocoann.py \
    --labelmap_path "$pred_dir" \
    --save_path /home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/"${dataset_dir}"/annotations/"$output_name".json
