#!/bin/sh
 
#SBATCH --job-name=cutlerInfer
#SBATCH --output=cutlerInfer-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=cutlerInfer-%A.err  # Standard error of the script
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=8  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=36G  # Memory in GB (Don't use more than 126G per GPU)
 
# activate corresponding environment
source ../../.venv/bin/activate

# navigate to the cutler directory
cd ../CutLER/cutler

# python demo/demo.py \
#     --config-file model_zoo/configs/CutLER-ImageNet/mask_rcnn_R_50_FPN.yaml \
#     --input ../../data/carotid-mini/images/train/*.jpg  --output outputs/inference/cutler_carotid_selftrain1_01 \
#     --opts MODEL.WEIGHTS outputs/full_carotid_selftrain-r1/model_final.pth

# cascade model
# python demo/demo.py \
#     --config-file model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml\
#     --input ../../data/carotid-mini/images/train/*.jpg  --output outputs/inference/cutler_carotid_cascade_selftrain1_finalmodel \
#     --opts MODEL.WEIGHTS outputs/cascade/full_carotid_selftrain-r1/model_final.pth

# CAROTID-MINI model
python demo/demo.py \
    --config-file model_zoo/configs/Carotid/cascade_mask_rcnn_R_50_FPN.yaml\
    --input ../../data/carotid-mini/images/train/*.jpg  --output outputs/inference/cutler_minicarotid_cascade_selftrain1_finalmodel \
    --opts MODEL.WEIGHTS outputs/cascade/carotid-mini/model_final.pth
    