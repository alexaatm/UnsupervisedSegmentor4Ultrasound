#!/bin/sh
 
#SBATCH --job-name=cutlerSelfTrain
#SBATCH --output=cutlerSelfTrain-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=cutlerSelfTrain-%A.err  # Standard error of the script
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=8  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=36G  # Memory in GB (Don't use more than 126G per GPU)
 
# activate corresponding environment
source ../../../.venv/bin/activate

# navigate to the cutler directory
cd ../../CutLER/cutler

export DETECTRON2_DATASETS=/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data

dataset_cutler_name=$1
model_weights=$2
output_name=$3
config=$4

# dataset=mutinfo_train_carotid_train_r1
# model_weights=http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_final.pth
# output_name="round1"

output_dir=outputs/self_train/"${dataset_cutler_name}"/"${output_name}/"
# create output directory
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

# mutinfo_train_carotid, self training round 1, initiate with imagenet cutler model
python train_net.py --num-gpus 1 \
  --config-file "${config}" \
  --train-dataset "${dataset_cutler_name}" \
  MODEL.WEIGHTS "$model_weights" \
  OUTPUT_DIR "$output_dir" # path to save checkpoints
  