#!/bin/sh
 
#SBATCH --job-name=maskcutConvert
#SBATCH --output=maskcutConvert-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=maskcutConvert-%A.err  # Standard error of the script
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=8  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=36G  # Memory in GB (Don't use more than 126G per GPU)
 
# activate corresponding environment
source ../../../.venv/bin/activate

# navigate to the cutler directory
cd ../../CutLER/cutler/tools

export DETECTRON2_DATASETS=/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data

dataset_path="US_MIXED/val"

# us mixed VAL maskcut pseudo masks
python cocoann_to_labelmap.py \
    --ann_path /home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/"${dataset_path}"/annotations/imagenet_train_fixsize480_tau0.15_N3.json \
    --save_path /home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/"${dataset_path}"/labelmaps/maskcut