param (
    [Parameter(Mandatory=$true, Position=0)]
    [string]$directory,

    [Parameter(Mandatory=$true, Position=1)]
    [string]$dataset
)

# Get the absolute path of the directory
$absolutePath = Convert-Path $directory

# Get a list of .ckpt files in the directory
$checkpoints = Get-ChildItem -Path $absolutePath -Filter "*.ckpt" -File

# Loop through each checkpoint and execute the Python command
foreach ($checkpoint in $checkpoints) {
    $checkpointPath = $checkpoint.FullName
    $checkpointStem = $checkpoint.BaseName

    # Build the Python command with the checkpoint stem, checkpoint path, and dataset
    $pythonCommand = "python -m pipeline.pipeline wandb.setup.project=pipelineVisTest wandb=defaults only_vis=False dataset=$dataset model=dino_vits8 loader=defaults precomputed=defaults vis.dino_attn_maps=True vis.eigen=True pipeline_steps.dino_features=True pipeline_steps.eigen=True model.checkpoint='$checkpointPath' model.label='$checkpointStem'"

    # Execute the Python command
    Invoke-Expression $pythonCommand
}
