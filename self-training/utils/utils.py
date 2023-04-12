import torch
from torchvision import transforms
from pathlib import Path
from models import dinoLightningModule, simclrLightningModule, simclrTripletLightningModule
import sys
import os
from torchsummary import summary
from torchinfo import summary as summary2

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

    if any(x in name for x in ('dino_vits16','dino_vits8','simclr')): #can add other models for which this transform makes sense
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

def get_image_paths(directory, image_extensions):
    """
    Returns a list of all image paths in the given directory.

    Parameters:
        directory (str): The path to the directory containing the images.

    Returns:
        image_paths (list of str): A list of all image paths in the directory.
    """
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(ext in file.lower() for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def get_file_paths_recursive(directory, file_extensions=None):
    """
    Returns a list of all file paths in the given directory and its subdirectories
    with the given file extensions. If file_extensions is None, all files are included.

    Parameters:
        directory (str): The path to the directory containing the files.
        file_extensions (list of str): A list of file extensions to include. If None, all files are included.

    Returns:
        file_paths (list of str): A list of all file paths in the directory and its subdirectories.
    """
    if file_extensions is None:
        file_extensions = []

    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(ext in file.lower() for ext in file_extensions) or len(file_extensions) == 0:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    return file_paths

def print_model_summary(backbone, model, h, w, ch = 3):
    if model=="simclr":
        if backbone=="resnet":
            backbone, hidden_dim = simclrLightningModule.get_resnet_backbone()
        else:
            raise NotImplementedError()
        model = simclrLightningModule.SimCLR(backbone,hidden_dim)
        s = summary(model.to('cpu'), input_size=(ch, h, w), device='cpu')

    elif model=="simclr_triplet":
        if backbone=="resnet":
            backbone, hidden_dim = simclrLightningModule.get_resnet_backbone()
        else:
            raise NotImplementedError()
        model = simclrTripletLightningModule.SimCLRTriplet(backbone,hidden_dim)
        s = summary(model.to('cpu'), input_size=[(ch, h, w),(ch, h, w),(ch, h, w)], device='cpu')
    
    elif model=="dino":
        if any(x in backbone for x in ('dino_vits16','dino_vits8')):
            backbone, input_dim = dinoLightningModule.get_dino_backbone(backbone)
        else:
            raise NotImplementedError()
        model = dinoLightningModule.DINO(backbone, input_dim)
        # use torchinfo summary instead of torchsummary (which doesnt work with this dino), but need to also pass batch size, eg 1
        s = summary2(model.to('cpu'), input_size=(1, ch, h, w), device='cpu')
    
    else:
        print("No such model considered in this project")
        raise NotImplementedError()

    # for liver-reduced-dataset, TODO: add as a dataset config where height and iwdth are defined
    # s = summary(model.to('cpu'), input_size=(3, 844, 648), device='cpu')
    # s = summary(model.to('cpu'), input_size=(3, 333, 256), device='cpu')

    print(s)

    # ref: https://stackoverflow.com/questions/46654424/how-to-calculate-optimal-batch-size
    print(""" (GPU_RAM - param_size) / (forward_back_ward_pass_size)
    Then round to powers of 2 ->  batch size""")

if __name__ == "__main__":
    backbone = sys.argv[1]
    model = sys.argv[2]
    h = int(sys.argv[3])
    w = int(sys.argv[4])
    print_model_summary(backbone, model, h, w)
