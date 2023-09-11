#!/bin/sh
 
#SBATCH --job-name=cutlerUnsupTrain
#SBATCH --output=cutlerUnsupTrain-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=cutlerUnsupTrain-%A.err  # Standard error of the script
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=8  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=10G  # Memory in GB (Don't use more than 126G per GPU)
 
# activate corresponding environment
source ../../.venv/bin/activate

# navigate to the cutler directory
cd ../CutLER/cutler

export DETECTRON2_DATASETS=/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data

dataset=$1
output_name=$2

# Example aprameters
## maskcut init
# dataset=mutinfo_train_carotid_main
# output_name=maskcut_init

## deep spectral init
# dataset=mutinfo_train_carotid_dsp_main
# output_name="dsp_init_clusters15_dino_ssd0_crf_segmaps"

# create output directory
output_dir=outputs/unsupervised_train/"${dataset}"/"${output_name}"/
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

python train_net.py --num-gpus 1 \
  --config-file model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml \
  --train-dataset "${dataset}" \
    OUTPUT_DIR "${output_dir}"