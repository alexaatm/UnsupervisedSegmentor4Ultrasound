import random
import torch
from torch.utils.data import Sampler
from torchvision.transforms import transforms
from datasets import datasets
from PIL import Image
import numpy as np

class PatchSampler(Sampler):
    def __init__(self, dataset, patch_size, patch_mode='random', shuffle=True):
        self.dataset = dataset
        self.patch_size = patch_size
        self.patch_mode = patch_mode
        self.shuffle=shuffle

        # generate patches
        if self.patch_mode=='grid':
            self.grid_sampler_init()
        elif self.patch_mode=='random':
            self.random_sampler_init()
        else:
            raise NotImplementedError()
        
        if self.shuffle:
            random.shuffle(self.indices)

    def grid_sampler_init(self):
        dataset_indices = []
        for idx in range(len(self.dataset)):
            image = self.dataset[idx][0]
            w, h = image.size[0], image.size[1]

            # find the crop of the image s.t. that is evenly divisible by the patch size
            crop_height = h - h % self.patch_size
            crop_width = w - w % self.patch_size
            num_patches_h = (crop_height - 1) // self.patch_size + 1
            num_patches_w = (crop_width - 1) // self.patch_size + 1

            # get all patches from image with index idx
            image_indices = [(idx, w*self.patch_size, h*self.patch_size, self.patch_size) for w in range(num_patches_w) for h in range(num_patches_h)]
            dataset_indices.append(image_indices)
        # concatenate all indices per all images into one list of indices
        self.indices = [i for image_indices in dataset_indices for i in image_indices]        

    def random_sampler_init(self):
        dataset_indices = []
        for idx in range(len(self.dataset)):
            image = self.dataset[idx][0]
            w, h = image.size[0], image.size[1]

            num_patches = int(h * w / (self.patch_size**2))

            # get random patches for a given image
            image_indices = [(idx, \
                              torch.randint(low = 0, high = w - self.patch_size + 1, size=(1,)).item(), \
                               torch.randint(low = 0, high = h - self.patch_size + 1, size=(1,)).item() , \
                                self.patch_size) for patch in range(num_patches)] 

            dataset_indices.append(image_indices)
        # concatenate all indices per all images into one list of indices
        self.indices = [i for image_indices in dataset_indices for i in image_indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class RandomPatchSampler(Sampler):
    def __init__(self, dataset, patch_size, shuffle=True):
        self.dataset = dataset
        self.patch_size = patch_size
        self.shuffle=shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        for idx in indices:
            image = self.dataset[idx][0]
            w, h = image.size[0], image.size[1]

            num_patches = int(h * w / (self.patch_size**2))

            # get random patches for a given image
            for patch in range(num_patches):
                yield (idx, \
                              torch.randint(low = 0, high = w - self.patch_size + 1, size=(1,)).item(), \
                               torch.randint(low = 0, high = h - self.patch_size + 1, size=(1,)).item() , \
                                self.patch_size)

    def __len__(self):
        image = self.dataset[0][0]
        w, h = image.size[0], image.size[1]
        num_patches_per_image = int(h * w / (self.patch_size**2))
        return len(self.dataset) * num_patches_per_image


if __name__ == "__main__":
    test_path="../data/liver2_mini/train"

    dataset=datasets.PatchDataset(root=test_path)

    sampler = RandomPatchSampler(dataset=dataset, patch_size=16)

    transform = transforms.Compose([transforms.ToTensor()])
    collate_fn = datasets.BaseCollateFunction(transform)

    # TODO: figure out how to combine sampler with shuffle, like if shufle is True, pick a random image to start with?
    dataloader =  torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=True,
        num_workers=1,
    )

    print(f'Dataset size:{len(dataset)}')
    print(f'Sampler size:{len(sampler)}')
    print(f'Dataloader size:{len(dataloader)}')
    # print(f'Dataloader shape:{dataloader.shape}')

    for batch in dataloader:
        im, label, _ = batch
        print(f'im[0]:{im[0].shape}, label[0]={label[0]}')
        break



