import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from extract import extract_utils as utils
from extract import extract
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import os


# A logger for this file
log = logging.getLogger(__name__)


def vis_eigen(cfg: DictConfig):
    dir=os.path.join(hydra.utils.get_original_cwd(),cfg.dataset.path)
    image_path=os.path.join(dir,cfg.sample_img_id+".jpg")
    im = np.array(Image.open(image_path))
    
    # eigs_path="./data/"+dataset+eigenvectors_dir+image_name+".pth"
    # feat_path="./data/"+dataset+features_dir+image_name+".pth"
    eigs_path = os.path.join(hydra.utils.get_original_cwd(),cfg.eigenvectors_dir, cfg.sample_img_id+".pth")
    feat_path = os.path.join(hydra.utils.get_original_cwd(),cfg.features_dir, cfg.sample_img_id+".pth")

    
    data_dict = torch.load(feat_path, map_location='cpu')
    data_dict.update(torch.load(eigs_path, map_location='cpu'))
    eigenvec_num = len(data_dict['eigenvectors'])
    eigenvectors = data_dict['eigenvectors'][:eigenvec_num].numpy()
    print(eigenvectors.shape)

    B, C, H, W, P, H_patch, W_patch, H_pad, W_pad = utils.get_image_sizes(data_dict)

    eigenvectors_img=eigenvectors.reshape(eigenvec_num, H_patch, W_patch)
   
    
    fig, ax = plt.subplots(nrows=2, ncols=eigenvec_num//2+1, figsize=(15, 7))
    for i, eigv_ax_pair in enumerate(zip(ax.flatten(),eigenvectors_img)):
      a, eigv = eigv_ax_pair
      a.imshow(eigv)
      a.title.set_text("eigv "+str(i))

    for a in ax.flatten(): 
      a.axis('off')
    plt.show()

    # visualize th eimage
    imgplot = plt.imshow(im)
    plt.title(cfg.sample_img_id)

@hydra.main(version_base=None, config_path="./configs", config_name="vis")
def run_experiment(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Current working directory  : {}".format(os.getcwd()))

    # wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    if cfg.experiment.name == "vis_eigen":
        log.info(f"Experiment chosen: {cfg.experiment.name}")
        # run = wandb.init(config=wandb_config, project = cfg.wandb.setup.project, settings=wandb.Settings(start_method='thread'))
        vis_eigen(cfg)
    else:
        raise ValueError(f'No experiment called: {cfg.experiment.name}')
    
    # wandb.finish()



if __name__ == "__main__":
    run_experiment()