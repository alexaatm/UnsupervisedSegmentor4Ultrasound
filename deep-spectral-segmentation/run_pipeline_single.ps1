param(
    [string]$num_segments,
    [string]$image_ssd_beta,
    # [string]$custom_path,
    [string]$image_dino_gamma,
    [string]$dataset
)

$custom_path_full_name="${dataset}/diff_cluster_n/clusters${num_segments}_dino${image_dino_gamma}_ssd${image_ssd_beta}"

$full_path = "C:\Users\Tmenova\personal\tum\thesis\thesis-codebase\deep-spectral-segmentation\outputs\$custom_path_full_name"


# Print information about the current job
Write-Host "Running deep spectral pipeline for $num_segments segments, image_ssd_beta=$image_ssd_beta, image_dino_gamma=$image_dino_gamma"

# Activate the corresponding environment (assuming it's a virtual environment)
& "C:/Users/Tmenova/personal/tum/thesis/thesis-codebase/thesis_env/Scripts/Activate"

# Navigate to the deep-spectral-segmentation directory
Set-Location -Path "C:\Users\Tmenova\personal\tum\thesis\thesis-codebase\deep-spectral-segmentation"

# For dataset - mutinfo train carotid
python -m pipeline.pipeline wandb.setup.project=pipelineVisTest wandb=defaults `
    only_vis=False  `
    dataset="$dataset" `
    model=dino_vits8 `
    loader=defaults `
    precomputed=defaults `
    vis.dino_attn_maps=False vis.eigen=False `
    vis.crf_segmaps=False vis.multiregion_segmaps=False `
    vis.segmaps=False  vis.crf_multi_region=False `
    pipeline_steps=allTrue `
    model.checkpoint="" `
    spectral_clustering.K="$num_segments" `
    multi_region_segmentation.non_adaptive_num_segments="$num_segments" `
    bbox.downsample_factor=8 `
    bbox.num_clusters="$num_segments" `
    crf.num_classes="$num_segments" `
    crf.downsample_factor=8 `
    spectral_clustering.image_ssd_beta="$image_ssd_beta" `
    custom_path_to_save_data="$full_path" `
    spectral_clustering.image_dino_gamma="$image_dino_gamma" `
    # pipeline_steps.dino_features=False `
    # crf.w1=10 crf.alpha=50 crf.beta=5 crf.w2=7 crf.gamma=3 crf.it=10
