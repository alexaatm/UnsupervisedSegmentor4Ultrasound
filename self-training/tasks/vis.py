import torch
import torchvision
import  torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import os

from lightly.data import LightlyDataset

from custom_utils import utils
from accelerate import Accelerator
from tqdm import tqdm

# A logger for this file
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="./configs", config_name="saliency_maps")
def extract_saliency_maps(cfg: DictConfig) -> None:
    """
    Based on: https://github.com/sunnynevarekar/pytorch-saliency-maps/blob/master/Saliency_maps_in_pytorch.ipynb
    """


    log.info(OmegaConf.to_yaml(cfg))
    log.info("Current working directory  : {}".format(os.getcwd()))

    # Transform
    # val_transform = utils.get_transform(cfg.model_name)
    # Add resize to the transforms
    if 'carotid' in cfg.dataset.name:
        # resize to acquare images (val set has varied sizes...)
        resize = transforms.Resize((cfg.dataset.input_size,cfg.dataset.input_size))
    else:
        resize = transforms.Resize(cfg.dataset.input_size)
    # transform = transforms.Compose([resize, val_transform])
    #define transforms to preprocess input image into format expected by model
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    #transforms to resize image to the size expected by pretrained model,
    #convert PIL image to tensor, and
    #normalize the image
    transform = transforms.Compose([
        resize,
        transforms.ToTensor(),
        normalize,          
    ])

    #inverse transform to get normalize image back to original form for visualization
    inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
    )


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

    # Model
    if cfg.model_checkpoint=="":
        model = torchvision.models.resnet50(pretrained=True)
    else:
        model_path = os.path.join(hydra.utils.get_original_cwd(), cfg.model_checkpoint)
        print("model path: ", model_path)
        model, _ = utils.get_model_from_path(cfg.model_name, model_path)

    #set model in eval mode
    model.eval()

    # get a test image
    sample, target, fname=dataset[0]
    print(sample)
    print(sample.shape)

    # Prepare accelerator
    cpu = True
    if torch.cuda.is_available():
        cpu = False
    accelerator = Accelerator(cpu)
    model = model.to(accelerator.device)
    print('accelerator devices=', accelerator.device)



    # Process
    pbar = tqdm(dataset, desc='Processing')
    for i, (sample, target, fname) in enumerate(pbar):   
        C, H, W = sample.shape
        print(f'sample shape: {sample.shape}')
        sample = sample.to(accelerator.device)

        input=sample.unsqueeze(0)

        #we want to calculate gradient of higest score w.r.t. input
        #so set requires_grad to True for input 
        input.requires_grad = True

        #forward pass to calculate predictions
        preds = model(input)
        score, indices = torch.max(preds, 1)
        #backward pass to get gradients of score predicted class w.r.t. input image
        score.backward()
        #get max along channel axiss
        slc,_ = torch.max(torch.abs(input.grad[0]), dim=0)
        #normalize to [0..1]
        slc = (slc - slc.min())/(slc.max()-slc.min())

        #apply inverse transform on image
        with torch.no_grad():
            input_img = inv_normalize(input[0])
            
        #plot image and its saleincy map
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(np.transpose(input_img.cpu().numpy(), (1, 2, 0)))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 2, 2)
        plt.imshow(slc.cpu().numpy())
        plt.xticks([])
        plt.yticks([])
        # plt.show()
        plt.savefig('saliency_map__'+fname)

if __name__ == "__main__":
    extract_saliency_maps()