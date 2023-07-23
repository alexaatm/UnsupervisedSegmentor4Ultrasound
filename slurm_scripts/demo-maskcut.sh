#!/bin/sh
 
#SBATCH --job-name=maskcut
#SBATCH --output=maskcut-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=maskcut-%A.err  # Standard error of the script
#SBATCH --time=0-24:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=8  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=20G  # Memory in GB (Don't use more than 126G per GPU)
 
# activate corresponding environment
source ../../.venv/bin/activate

# navigate to the cutler directory
cd ../CutLER/maskcut
 
# SINGLE IMAGE
# python demo.py \
#     --img-path ../../data/carotid-mini/images/train/img0001.jpg \
#     --N 3 --tau 0.15 --vit-arch base --patch-size 8 --fixed_size 480

# DIRECTORY
python demo.py \
    --input-dir ../../data/carotid-mini/images/train/ \
    --N 3 --tau 0.15 --vit-arch base --patch-size 8 --fixed_size 480 \
    --output_path outputs/caortid-mini_N3_tau0_15_fs480