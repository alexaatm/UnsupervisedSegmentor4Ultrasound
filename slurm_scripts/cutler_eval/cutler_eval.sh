#!/bin/sh
 
#SBATCH --job-name=curltEval
#SBATCH --output=curltEval-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=curltEval-%A.err  # Standard error of the script
#SBATCH --gres=gpu:0  # Number of GPUs if needed
#SBATCH --cpus-per-task=8  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=4G  # Memory in GB (Don't use more than 126G per GPU) BEFORE: 36G
 
# activate corresponding environment
source ../../../.venv/bin/activate

# navigate to the main directory
cd ../../

export DETECTRON2_DATASETS=/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data

dataset=$1
pred_dir=$2
tag=$3

eval_per_image_values=(True) # (True False)

echo dataset="${dataset}"
echo pred_dir="${pred_dir}"
echo eval_per_image_values="${eval_per_image_values}"
echo tag="${tag}"


echo "EVALUATION..."
# setup wandb to work in the node
export WANDB_API_KEY=163fac3be1f95e6eeb5964f1743469286a0421ae
export WANDB_CONFIG_DIR=/tmp/
export WANDB_DIR=/tmp/
export WANDB_CACHE_DIR=/tmp/

for eval_per_image in "${eval_per_image_values[@]}"; do
    python -m evaluation.segm_eval \
      dataset="${dataset}" \
      vis_rand_k=20 \
      iou_thresh=0.0 \
      eval_per_image="$eval_per_image" \
      dataset.pred_dir="$pred_dir" \
      wandb.tag="$tag"
done