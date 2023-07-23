#!/bin/sh
 
#SBATCH --job-name=fullCarotidMask
#SBATCH --output=fullCarotidMask-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=fullCarotidMask-%A.err  # Standard error of the script
#SBATCH --time=0-24:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=8  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=20G  # Memory in GB (Don't use more than 126G per GPU)
 
# activate corresponding environment
source ../../.venv/bin/activate

# navigate to the cutler directory
cd ../CutLER/maskcut
 
# run the program
python3 run_with_submitit_maskcut_array.py \
    --ngpus 1 \
    --nodes 1 \
    --timeout 1200 \
    --partition part-1 \
    --vit-arch base \
    --patch-size 8 \
    --dataset-path ../../data/full_carotid/images/train \
    --tau 0.15 \
    --out-dir ../../data/full_carotid/annotations_per_folder \
    --num-folder-per-job 1\
    --job-index 0 \
    --fixed_size 480 \
    --N 3 
    
# python maskcut.py \
#     --vit-arch base --patch-size 8 \
#     --tau 0.15 --fixed_size 480 --N 3 \
#     --dataset-path ../../data/carotid_mutinfo/images/train \
#     --out-dir ../../data/carotid_mutinfo/annotations/ \
