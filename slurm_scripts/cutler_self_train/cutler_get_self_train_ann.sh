#!/bin/sh
 
#SBATCH --job-name=cutlerPrGetAnn
#SBATCH --output=cutlerPrGetAnn-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=cutlerPrGetAnn-%A.err  # Standard error of the script
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=8  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=36G  # Memory in GB (Don't use more than 126G per GPU)
 
# activate corresponding environment
source ../../../.venv/bin/activate

# navigate to the cutler directory
cd ../../CutLER/cutler

export DETECTRON2_DATASETS=/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data

dataset_dir=$1
dataset_cutler_name=$2
model_weights=$3
thresh=$4
pred_name=$5
config=$6

output_pred_name="${pred_name}"_thresh"${thresh}"
output_pred_dir=outputs/inference/"${dataset_cutler_name}"/"${output_pred_name}"

echo dataset_dir="${dataset_dir}"
echo dataset_cutler_name="${dataset_cutler_name}"
echo thresh="${thresh}"
echo output_pred_dir="${output_pred_dir}"

# predict on TRAIN data
# create output directory
if [ ! -d "$output_pred_dir" ]; then
    mkdir -p "$output_pred_dir"
fi

# echo "PREDICTION..."
# python train_net.py --num-gpus 1 \
#   --config-file "${config}" \
#   --test-dataset  "${dataset_cutler_name}" \
#   --eval-only TEST.DETECTIONS_PER_IMAGE 15 \
#   MODEL.WEIGHTS "$model_weights" \
#   OUTPUT_DIR "$output_pred_dir"\

# navigate to tools directory
cd tools


# us mixed maskcu cut init unsupervised model  - convert to cutler ann for self training
python get_self_training_ann.py \
 --new-pred /home/guests/oleksandra_tmenova/test/project/thesis-codebase/CutLER/cutler/outputs/inference/"${dataset_cutler_name}"/"${output_pred_name}"/inference/coco_instances_results.json \
 --prev-ann /home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/"${dataset_dir}"/annotations/imagenet_train_fixsize480_tau0.15_N3.json \
 --save-path /home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/"${dataset_dir}"/annotations/cutler_"${output_pred_name}"_r1.json \
 --threshold "${thresh}"