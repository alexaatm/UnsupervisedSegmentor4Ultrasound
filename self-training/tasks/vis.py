import torch
from torch import nn
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

import ssl

# A logger for this file
log = logging.getLogger(__name__)

def extract_saliency_maps_v1_avg(cfg: DictConfig) -> None:
    """
    Based on: https://github.com/sunnynevarekar/pytorch-saliency-maps/blob/master/Saliency_maps_in_pytorch.ipynb
    """

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
        sample = sample.to(accelerator.device)

        input=sample.unsqueeze(0)

        #we want to calculate gradient of higest score w.r.t. input
        #so set requires_grad to True for input 
        input.requires_grad = True

        #forward pass to calculate predictions
        preds = model(input)
        print(f'preds shape: {preds.shape}')
        preds=preds.squeeze()
        print(f'preds shape squeezed: {preds.shape}')

        # works for feature sixe 512, for 1000 breaks cause out of memory
        slc_maps = []
        for p in preds:
            p.backward(retain_graph=True)
            slc,_ = torch.max(torch.abs(input.grad[0]), dim=0)
            slc = (slc - slc.min())/(slc.max()-slc.min())
            slc_maps.append(slc)
            p.detach()

        stacked_slc_maps = torch.stack(slc_maps)
        preds = (preds - preds.min())/(preds.max()-preds.min())
        weighted_slc_maps= stacked_slc_maps * preds[:, None, None]
        # Take the mean along the first dimension
        slc = torch.mean(weighted_slc_maps, dim=0)

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
        plt.imshow(slc.detach().cpu().numpy())
        plt.xticks([])
        plt.yticks([])
        # plt.show()
        plt.savefig('saliency_map_v1__'+fname)

def extract_saliency_maps_v1(cfg: DictConfig) -> None:
    """
    Based on: https://github.com/sunnynevarekar/pytorch-saliency-maps/blob/master/Saliency_maps_in_pytorch.ipynb
    """

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
        sample = sample.to(accelerator.device)

        input=sample.unsqueeze(0)

        #we want to calculate gradient of higest score w.r.t. input
        #so set requires_grad to True for input 
        input.requires_grad = True

        #forward pass to calculate predictions
        preds = model(input)
        print(f'preds shape: {preds.shape}')
        score, indices = torch.max(preds, 1)
        print(f'score shape: {score.shape}')

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
        plt.savefig('saliency_map_v1__'+fname)

def extract_saliency_maps_v2(cfg: DictConfig) -> None:
    """
    Based on: Registering a hook to get output of the last convolutional layer
    """

    # Transform
    # Add resize to the transforms
    if 'carotid' in cfg.dataset.name:
        # resize to acquare images (val set has varied sizes...)
        resize = transforms.Resize((cfg.dataset.input_size,cfg.dataset.input_size))
    else:
        resize = transforms.Resize(cfg.dataset.input_size)
    # val_transform = utils.get_transform(cfg.model_name)
    # transform = transforms.Compose([resize, val_transform])

    #define transforms to preprocess input image into format expected by model
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

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
    log.info(f'Dataset size: {len(dataset)}')

    # Model
    if cfg.model_checkpoint=="":
        resnet = torchvision.models.resnet50(pretrained=True)
        model = nn.Sequential(*list(resnet.children())[:-1])
        # ssl._create_default_https_context = ssl._create_unverified_context
        # model = torch.hub.load('facebookresearch/dino:main', cfg.model_name, pretrained=True) 
    else:
        model_path = os.path.join(hydra.utils.get_original_cwd(), cfg.model_checkpoint)
        print("model path: ", model_path)
        model, params = utils.get_model_from_path(cfg.model_name, model_path)

    #set model in eval mode
    model.eval()

    # register a hook
    if 'dino' in cfg.model_name:
        # hook
        which_block = -1
        feat_out = {}
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output
        model._modules["blocks"][which_block]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
        num_heads = params[0]
        patch_size = params[1]
    elif 'simclr' in cfg.model_name:
        feat_out = {}
        # define hook function
        def hook(module, input, output):
            # store the output of the module in the outputs dictionary
            feat_out['last_conv_layer_input'] = input
        # register hook on the last convolutional layer of the resnet model
        last_layer = list(model.children())[-1]
        last_layer.register_forward_hook(hook)

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
    print('accelerator device=', accelerator.device)

    # Process
    pbar = tqdm(dataset, desc='Processing')
    for i, (sample, target, fname) in enumerate(pbar):   
        C, H, W = sample.shape
        sample = sample.to(accelerator.device)

        input=sample.unsqueeze(0)
        output = model(input)

        last_conv_layer_input=feat_out['last_conv_layer_input'][0]
        print(f'shape of last_conv_layer: {last_conv_layer_input.shape}')

        summed_features,_ = torch.max(last_conv_layer_input, dim=1)
        print(f'summed_features shape: {summed_features.shape}')

        upscaled_features=torch.nn.functional.interpolate(summed_features.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
        print(f'output shape: {upscaled_features.shape}')

        slc = upscaled_features[0]

        #apply inverse transform on image
        input_img = inv_normalize(input[0])
            
        #plot image and its saleincy map
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(np.transpose(input_img.cpu().numpy(), (1, 2, 0)))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 2, 2)
        plt.imshow(slc.detach().cpu().numpy())
        plt.xticks([])
        plt.yticks([])
        # plt.show()
        plt.savefig('saliency_map_v2__'+fname)

def extract_saliency_maps_dino(cfg: DictConfig) -> None:
    """
    Based on: Registering a hook to get output of the last convolutional layer
    """

    # Transform
    # Add resize to the transforms
    if 'carotid' in cfg.dataset.name:
        # resize to acquare images (val set has varied sizes...)
        resize = transforms.Resize((cfg.dataset.input_size,cfg.dataset.input_size))
    else:
        resize = transforms.Resize(cfg.dataset.input_size)
    # val_transform = utils.get_transform(cfg.model_name)
    # transform = transforms.Compose([resize, val_transform])

    #define transforms to preprocess input image into format expected by model
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

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
    log.info(f'Dataset size: {len(dataset)}')

    # Model
    if cfg.model_checkpoint=="":
        ssl._create_default_https_context = ssl._create_unverified_context
        model, params = utils.get_dino_traind_model(cfg.model_name)
        # model = torch.hub.load('facebookresearch/dino:main', cfg.model_name, pretrained=True) 
    else:
        model_path = os.path.join(hydra.utils.get_original_cwd(), cfg.model_checkpoint)
        print("model path: ", model_path)
        model, params = utils.get_model_from_path(cfg.model_name, model_path)

    #set model in eval mode
    model.eval()

    # register a hook
    if 'dino' in cfg.model_name:
        # hook
        which_block = -1
        feat_out = {}
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output
        model._modules["blocks"][which_block]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
        num_heads = params[0]
        patch_size = params[1]

    # Prepare accelerator
    cpu = True
    if torch.cuda.is_available():
        cpu = False
    accelerator = Accelerator(cpu)
    model = model.to(accelerator.device)
    print('accelerator device=', accelerator.device)

    # Process
    pbar = tqdm(dataset, desc='Processing')
    for i, (sample, target, fname) in enumerate(pbar):   
        sample=sample.unsqueeze(0)
        print(f'sample shape before: {sample.shape}')
        B, C, H, W = sample.shape
        P = patch_size
        print(f'patch size: {P}')
        H_patch, W_patch = H // P, W // P
        H_pad, W_pad = H_patch * P, W_patch * P
        T = H_patch * W_patch + 1  # number of tokens, add 1 for [CLS]
        print(f'Tokens num : {T}')
        
        sample = sample[:, :, :H_pad, :W_pad]
        sample = sample.to(accelerator.device)
        print(f'sample shape: {sample.shape}')
        
        # extarct features
        model.get_intermediate_layers(sample)[0].squeeze(0)
        output_qkv = feat_out["qkv"].reshape(B, T, 3, num_heads, -1 // num_heads).permute(2, 0, 3, 1, 4)
        print(f'output_qkv shape: {output_qkv.shape}')
        slc = output_qkv[1].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
        print(f'slc shape: {slc.shape}')
        #apply inverse transform on image
        input_img = inv_normalize(sample[0])
        print(f'input_img shape: {input_img.shape}')

        
            
        #plot image and its saleincy map
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(np.transpose(input_img.cpu().numpy(), (1, 2, 0)))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 2, 2)
        plt.imshow(np.transpose(slc.cpu().numpy(), (1, 2, 0)))
        plt.xticks([])
        plt.yticks([])
        plt.show()
        exit()
        plt.savefig('saliency_map_v2__'+fname)

@hydra.main(version_base=None, config_path="./configs", config_name="saliency_maps")
def vis(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Current working directory  : {}".format(os.getcwd()))

    log.info(f"Visualisation chosen: {cfg.vis}")
    if cfg.vis == "saliency_maps_v1":
        extract_saliency_maps_v1(cfg)
    elif cfg.vis == "saliency_maps_v2":
        extract_saliency_maps_v2(cfg)
    elif cfg.vis == "saliency_maps_v1_avg":
        extract_saliency_maps_v1_avg(cfg)
    elif cfg.vis == "saliency_maps_dino":
        extract_saliency_maps_dino(cfg)
    else:
        raise ValueError(f'No visualisation called: {cfg.vis}')

if __name__ == "__main__":
    vis()