cd deep-spectral-segmentation

export WANDB_API_KEY=<your-wandb-api>
export WANDB_CONFIG_DIR=/tmp/
export WANDB_CACHE_DIR=/tmp/
export WANDB_AGENT_MAX_INITIAL_FAILURE=20
export WANDB__SERVICE_WAIT=600
export XFORMERS_DISABLED=True

python -m pipeline.pipeline_sweep_subfolders \
    vis=selected \
    pipeline_steps=defaults \
    dataset=thyroid \
    wandb.tag=test \
    sweep=defaults