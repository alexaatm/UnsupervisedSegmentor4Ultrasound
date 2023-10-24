#!/bin/sh
 
#SBATCH --job-name=cutlrPr2L
#SBATCH --output=cutlrPr2L-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=cutlrPr2L-%A.err  # Standard error of the script
#SBATCH --gres=gpu:0  # Number of GPUs if needed
#SBATCH --cpus-per-task=8  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=4G  # Memory in GB (Don't use more than 126G per GPU) BEFORE: 36G
 
# activate corresponding environment
source ../../.venv/bin/activate

# navigate to the cutler directory
cd ../CutLER/cutler

export DETECTRON2_DATASETS=/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data

dataset=$1
model_weights=$2
thresh=$3
pred_name=$4
tag=$5
config=$6

# dataset="mutinfo_val_carotid"
# model_weights="/home/guests/oleksandra_tmenova/test/project/thesis-codebase/CutLER/cutler/outputs/self_train/mutinfo_train_carotid/round1/model_final.pth"
# thresh=0.35
# output_pred_name=cutler_imagenet_selftrain_r1_model_final_thresh"${thresh}"
output_pred_name="${pred_name}"_thresh"${thresh}"
output_pred_dir=outputs/inference/"${dataset}"/"${output_pred_name}"
eval_per_image_values=(True) # (True False)
# tag='cutler'
# config=model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN_demo.yaml

echo dataset="${dataset}"
echo thresh="${thresh}"
echo pred_name="${pred_name}"
echo output_pred_name="${output_pred_name}"
echo output_pred_dir="${output_pred_dir}"
echo eval_per_image_values="${eval_per_image_values}"
echo tag="${tag}"

# create output directory
if [ ! -d "$output_pred_dir" ]; then
    mkdir -p "$output_pred_dir"
fi

echo "PREDICTION..."
# mutinfo_val_carotid_main - predict using cutler imagenet model SELF TRAINED on mutinfo_train_carotid_train_r1,  checkpoint final
python train_net.py --num-gpus 1 \
  --config-file "${config}" \
  --test-dataset  "${dataset}"_main \
  --eval-only TEST.DETECTIONS_PER_IMAGE 15 \
  MODEL.WEIGHTS "$model_weights" \
  OUTPUT_DIR "$output_pred_dir"\

# navigate to tools dorectory
cd tools

# convert round1 train predict to cutler annotation
echo "CONVERSION TO CUTLER ANN..."
python predann_to_cutlerann.py \
  --new-pred /home/guests/oleksandra_tmenova/test/project/thesis-codebase/CutLER/cutler/"${output_pred_dir}"/inference/coco_instances_results.json \
  --cutler-ann /home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/"${dataset}"/annotations/imagenet_train_fixsize480_tau0.15_N3.json \
  --save-path /home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/"${dataset}"/annotations/"${output_pred_name}".json \
  --threshold "$thresh"

# convert to labelmaps
echo "CONVERSION TO LABELMAPS..."
python cocoann_to_labelmap.py \
  --ann_path /home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/"${dataset}"/annotations/"${output_pred_name}".json \
  --save_path /home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/"${dataset}"/labelmaps/"${output_pred_name}"

# navigate to the cutler directory
cd ../../../evaluation

echo "EVALUATION..."
# setup wandb to work in the node
export WANDB_API_KEY=163fac3be1f95e6eeb5964f1743469286a0421ae
export WANDB_CONFIG_DIR=/tmp/
export WANDB_DIR=/tmp/
export WANDB_CACHE_DIR=/tmp/

for eval_per_image in "${eval_per_image_values[@]}"; do
    python segm_eval.py \
      dataset="$dataset" \
      vis_rand_k=20 \
      iou_thresh=0.0 \
      eval_per_image="$eval_per_image" \
      dataset.pred_dir=/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/"$dataset"/labelmaps/"${output_pred_name}" \
      wandb.tag="$tag"
done