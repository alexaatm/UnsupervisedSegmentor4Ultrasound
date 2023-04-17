import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
from custom_utils import utils
import torch
from lightly.data import LightlyDataset
import os
from accelerate import Accelerator
from tqdm import tqdm
from torchvision import transforms

# A logger for this file
log = logging.getLogger(__name__)


def extract_features(cfg: DictConfig) -> None:
    # adapted from https://github.com/lukemelas/deep-spectral-segmentation/tree/main/semantic-segmentation
    
    
    # Output
    utils.make_output_dir(cfg.output_dir)
    
    # Models
    model_name = cfg.model_name.lower()
    model_path = os.path.join(hydra.utils.get_original_cwd(), cfg.model_checkpoint)
    print("model path: ", model_path)
    model, params = utils.get_model_from_path(model_name, model_path)
    val_transform = utils.get_transform(model_name)

    # Add resize to the transforms
    if 'carotid' in cfg.dataset.name:
        # resize to acquare images (val set has varied sizes...)
        resize = transforms.Resize((cfg.dataset.input_size,cfg.dataset.input_size))
    else:
        resize = transforms.Resize(cfg.dataset.input_size)
    transform = transforms.Compose([resize, val_transform])

    # Add dino_spceific params - hook and numheads
    if 'dino' in model_name:
        # hook
        which_block = -1
        feat_out = {}
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output
        model._modules["blocks"][which_block]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
        num_heads = params[0]
        patch_size = params[1]

    # Dataset
    dataset = LightlyDataset(
        input_dir = os.path.join(hydra.utils.get_original_cwd(),cfg.dataset.path),
        transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers)
    log.info(f'Dataset size: {len(dataset)}')
    log.info(f'Dataloader size: {len(dataloader)}')

    # Prepare accelerator
    cpu = True
    if torch.cuda.is_available():
        cpu = False
    accelerator = Accelerator(cpu)
    model = model.to(accelerator.device)

    # Process
    pbar = tqdm(dataloader, desc='Processing')
    for i, (samples, targets, fnames) in enumerate(pbar):   
        output_dict = {}

        # Check if file already exists
        id = Path(fnames[0]).stem
        output_file = Path(cfg.output_dir) / f'{id}.pth'
        if output_file.is_file():
            pbar.write(f'Skipping existing file {str(output_file)}')
            continue

        B, C, H, W = samples.shape
        # print(f'samples shape: {samples.shape}')       

        # Forward and collect features into output dict
        if 'dino' in model_name:
            # reshape image
            P = patch_size
            H_patch, W_patch = H // P, W // P
            H_pad, W_pad = H_patch * P, W_patch * P
            T = H_patch * W_patch + 1  # number of tokens, add 1 for [CLS]
            samples = samples[:, :, :H_pad, :W_pad]
            samples = samples.to(accelerator.device)

            # extarct features
            model.get_intermediate_layers(samples)[0].squeeze(0)
            output_qkv = feat_out["qkv"].reshape(B, T, 3, num_heads, -1 // num_heads).permute(2, 0, 3, 1, 4)
            output_dict['k'] = output_qkv[1].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
            output_dict['patch_size'] = patch_size
            output_dict['shape'] = (B, C, H, W)
        
        elif 'simclr' in model_name:
            samples = samples.to(accelerator.device)
            output_dict['simclr'] = model(samples).flatten(start_dim=1)
            output_dict['shape'] = (output_dict['simclr'].shape)
        else:
            raise ValueError(model_name)

        # Metadata
        output_dict['file'] = fnames[0]
        output_dict['id'] = id
        output_dict['model_name'] = model_name
        
        output_dict = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in output_dict.items()}

        # Save
        accelerator.save(output_dict, str(output_file))
        accelerator.wait_for_everyone()
    
    log.info(f'Saved features to {cfg.output_dir}')

@hydra.main(version_base=None, config_path="./configs", config_name="features")
def run_experiment(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Current working directory  : {}".format(os.getcwd()))

    if cfg.experiment.name == "extract_features":
        log.info(f"Experiment chosen: {cfg.experiment.name}")
        extract_features(cfg)
    else:
        raise ValueError(f'No experiment called: {cfg.experiment.name}')



if __name__ == "__main__":
    run_experiment()