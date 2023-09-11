#!/bin/sh
 
#SBATCH --job-name=dspSinglePipeline
#SBATCH --output=dspSinglePipeline-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=dspSinglePipeline-%A.err  # Standard error of the script
#SBATCH --time=0-24:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --mem=10G  # Memory in GB (Don't use more than 126G per GPU)
#SBATCH --cpus-per-task=8  # Number of CPUs (Don't use more than 24 per GPU)
 
# arguments
num_segments=$1
image_ssd_beta=$2
custom_path=$3

full_path="/home/guests/oleksandra_tmenova/test/project/thesis-codebase/deep-spectral-segmentation/outputs/${custom_path}"

# activate corresponding environment
source ../../.venv/bin/activate

# navigate to the cutler directory
cd ../deep-spectral-segmentation

# Print infromation about the current job
echo "Running deep spectral pipeline for $num_segments segments, image_ssd_beta=$image_ssd_beta"

# For dataset - mutinfo train carotid
python -m pipeline.pipeline wandb.setup.project=pipelineVisTest wandb=defaults \
    only_vis=False  \
    dataset=mutinfo_train_carotid \
    model=dino_vits8 \
    loader=defaults \
    precomputed=defaults \
    vis.dino_attn_maps=False vis.eigen=False \
    vis.crf_segmaps=False vis.multiregion_segmaps=False \
    vis.segmaps=False  vis.crf_multi_region=False \
    pipeline_steps=allTrue \
    model.checkpoint="" \
    spectral_clustering.K="$num_segments" \
    multi_region_segmentation.non_adaptive_num_segments="$num_segments" \
    bbox.downsample_factor=8 \
    bbox.num_clusters="$num_segments" \
    crf.num_classes="$num_segments" \
    crf.downsample_factor=8 \
    spectral_clustering.image_ssd_beta="$image_ssd_beta" \
    custom_path_to_save_data="$full_path" 
    # crf.w1=10 crf.alpha=50 crf.beta=5 crf.w2=7 crf.gamma=3 crf.it=10 \
