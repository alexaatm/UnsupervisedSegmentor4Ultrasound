#!/bin/sh
 
#SBATCH --job-name=cutlertrain
#SBATCH --output=cutlertrain-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=cutlertrain-%A.err  # Standard error of the script
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=8  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=36G  # Memory in GB (Don't use more than 126G per GPU)
 
# activate corresponding environment
source ../../.venv/bin/activate

# navigate to the cutler directory
cd ../CutLER/cutler

# set the path to datasets
export DETECTRON2_DATASETS=../../data/
 
# run the program
# python train_net.py --num-gpus 1 \
#   --config-file model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml \
#   --test-dataset full_carotid_train \
#   --eval-only TEST.DETECTIONS_PER_IMAGE 30 \ 


# CASCADE FULL CAROTID - wrong config-weights combo
# python train_net.py --num-gpus 1 \
#     --config-file model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN_self_train.yaml \
#     --train-dataset full_carotid_train \
#     MODEL.WEIGHTS http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_final.pth  \
#     OUTPUT_DIR outputs/cascade/full_carotid_selftrain-r1 \
#     # MODEL.WEIGHTS http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_mrcnn_final.pth \

# CASCADE FULL CAROTID - diff config (non self train)
python train_net.py --num-gpus 1 \
    --config-file model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml \
    --train-dataset full_carotid_train \
    MODEL.WEIGHTS http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_final.pth  \
    OUTPUT_DIR outputs/cascade/full_carotid_selftrain-r1_imagenetconfig 


# CASCADE CAROTID-MINI
# python train_net.py --num-gpus 1 \
#     --config-file model_zoo/configs/Carotid/cascade_mask_rcnn_R_50_FPN.yaml \
#     --train-dataset carotid-mini_train \
#     OUTPUT_DIR outputs/cascade/carotid-mini \

