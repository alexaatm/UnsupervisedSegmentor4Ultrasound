import torch
from torchvision import transforms
from pathlib import Path
from models import dinoLightningModule, simclrLightningModule
import sys

def get_model(name: str):
    # https://github.com/lukemelas/deep-spectral-segmentation/tree/main/semantic-segmentation

    if 'dino' in name:
        model = torch.hub.load('facebookresearch/dino:main', name)
        model.fc = torch.nn.Identity()
        val_transform = get_transform(name)
        patch_size = model.patch_embed.patch_size
        num_heads = model.blocks[0].attn.num_heads
    else:
        raise ValueError(f'Cannot get model: {name}')
    model = model.eval()
    return model, val_transform, patch_size, num_heads


def get_transform(name: str):
    # https://github.com/lukemelas/deep-spectral-segmentation/tree/main/semantic-segmentation

    if any(x in name for x in ('dino_vits16')): #can add other models for which this transform makes sense
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transform = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        raise NotImplementedError()
    return transform

def make_output_dir(output_dir, check_if_empty=True):
    # https://github.com/lukemelas/deep-spectral-segmentation/tree/main/semantic-segmentation
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    if check_if_empty and (len(list(output_dir.iterdir())) > 0):
        print(f'Output dir: {str(output_dir)}')
        if input(f'Output dir already contains files. Continue? (y/n) >> ') != 'y':
            sys.exit()  # skip because already generated

def get_model_from_path(model_name, ckpt_path):
    if 'dino' in model_name:
        # get the backbone
        backbone, input_dim = dinoLightningModule.get_dino_backbone(model_name)
        patch_size = backbone.patch_embed.patch_size
        num_heads = backbone.blocks[0].attn.num_heads
        backbone.fc = torch.nn.Identity() # why do we need to set it to identity?
        

        # load the model from the checkpoint
        checkpoint = torch.load(ckpt_path)
        print(checkpoint.keys())
        state_dict = checkpoint['state_dict']
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}


        full_model = dinoLightningModule.DINO(backbone, input_dim)
        full_model.load_state_dict(state_dict, strict=False)

        val_transform = get_transform(model_name)

        # take teacher backbone as a model for inference
        model = full_model.teacher_backbone
    # TODO: add elif branch for loading simclr model
    else:
        raise ValueError(f'Cannot get model: {model_name}')
    model = model.eval()
    return model, val_transform, patch_size, num_heads