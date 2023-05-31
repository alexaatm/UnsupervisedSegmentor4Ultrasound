import hydra
from omegaconf import DictConfig, OmegaConf
import os
import logging
import wandb

from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path

from extract import extract


# A logger for this file
log = logging.getLogger(__name__)

# Pipeline for deep spectral segmentation steps
def pipeline(cfg: DictConfig) -> None:
    log.info("Starting the pipeline...")


    # Extract Features
    log.info("STEP 1/8: extract features")


    # Set the directories
    if cfg.wandb.mode=='server':
        # use polyaxon paths
        main_data_dir = os.path.join(get_data_paths()['data1'], '3D_US_vis', 'datasets')
        path_to_save_data = os.path.join(get_outputs_path(), cfg.dataset.name)
    else:
        # use default local data
        main_data_dir = os.path.join(hydra.utils.get_original_cwd(), '../data')
        path_to_save_data = os.path.join(os.getcwd())


    images_list = os.path.join(main_data_dir, cfg.dataset.name, 'lists', 'images.txt')
    images_root = os.path.join(main_data_dir, cfg.dataset.name, 'images')
    output_dir = os.path.join(path_to_save_data, 'features', cfg.model.name)

    log.info(f'images_list={images_list}')
    log.info(f'images_root={images_root}')
    log.info(f'output_dir for features={output_dir}')


    extract.extract_features(
        images_list=images_list,
        images_root=images_root,
        output_dir=output_dir,
        model_name=cfg.model.name,
        batch_size=cfg.loader.batch_size
    )



    # Compute Eigenvectors - spectral clustering step


    # Extract segments


    # Extract bounding boxes


    # Extract bounding box features


    # Extract clusters


    # Create semantic segmentations


    # Create crf segmentations (optional)

    
@hydra.main(version_base=None, config_path="./configs", config_name="defaults")
def run_pipeline(cfg: DictConfig) -> None:
    print(f'cfg.wandb.mode is={cfg.wandb.mode}')

    if cfg.wandb.mode=='server':
        # login to wandb using locally stored key, remove the key to prevent it from being logged
        wandb.login(key=cfg.wandb.key)
        cfg.wandb.key=""
        
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Current working directory  : {}".format(os.getcwd()))

    # wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # run = wandb.init(config=wandb_config, project = cfg.wandb.setup.project, settings=wandb.Settings(start_method='thread'))
    
    pipeline(cfg)    

    # wandb.finish()



if __name__ == "__main__":
    run_pipeline()