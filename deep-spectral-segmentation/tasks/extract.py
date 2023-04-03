import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from extract import extract
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import os


# A logger for this file
log = logging.getLogger(__name__)


def extract_eigen(cfg: DictConfig):

    extract.extract_eigs(
        images_root = os.path.join(hydra.utils.get_original_cwd(),cfg.dataset.path),
        features_dir = os.path.join(hydra.utils.get_original_cwd(),cfg.features_dir),
        output_dir = cfg.output_dir,
        K = cfg.K,
        image_color_lambda = cfg.image_color_lambda,
        image_ssd_beta = cfg.image_ssd_beta,
        multiprocessing = cfg.multiprocessing
    )

@hydra.main(version_base=None, config_path="./configs", config_name="eigen")
def run_experiment(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Current working directory  : {}".format(os.getcwd()))

    # wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    if cfg.experiment.name == "extract_eigen":
        log.info(f"Experiment chosen: {cfg.experiment.name}")
        # run = wandb.init(config=wandb_config, project = cfg.wandb.setup.project, settings=wandb.Settings(start_method='thread'))
        extract_eigen(cfg)
    # TODO: add hydra wrappers for all functions from deep-spectral
    else:
        raise ValueError(f'No experiment called: {cfg.experiment.name}')
    
    # wandb.finish()



if __name__ == "__main__":
    run_experiment()