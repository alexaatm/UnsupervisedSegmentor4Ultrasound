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


class TripletPatchSampler(Sampler):
    def __init__(self, dataset, patch_size, shuffle=True, max_shift=30):
        self.dataset = dataset
        self.patch_size = patch_size
        self.shuffle=shuffle
        self.max_shift=max_shift    


    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        for idx in indices:
            image = self.dataset[idx][0]
            w, h = image.size[0], image.size[1]

            num_patches = int(h * w / (self.patch_size**2))

            # get random patches for a given image
            for i in range(num_patches):
                # sample anchor patch
                anchor_patch = (torch.randint(low = 0, high = w - self.patch_size + 1, size=(1,)).item(), \
                               torch.randint(low = 0, high = h - self.patch_size + 1, size=(1,)).item())
                
                # sample positive patch by shifting range from the anchor
                w_shift = torch.randint(low = 0, high = self.max_shift, size=(1,)).item()
                h_shift = torch.randint(low = 0, high = self.max_shift, size=(1,)).item()
                pos_patch = (torch.randint(low = max(anchor_patch[0] - w_shift, 0),
                                           high = min(anchor_patch[0] + self.patch_size//2 + w_shift, w),
                                           size=(1,)).item(), \
                             torch.randint(low = max(anchor_patch[1] - h_shift, 0),
                                           high = min(anchor_patch[1] + self.patch_size//2 + w_shift, h),
                                           size=(1,)).item())
                
                # sample negative randomly (TODO: add non random negative sampling)
                neg_patch = (torch.randint(low = 0, high = w - self.patch_size + 1, size=(1,)).item(), \
                               torch.randint(low = 0, high = h - self.patch_size + 1, size=(1,)).item())
                yield (idx, anchor_patch, pos_patch, neg_patch, self.patch_size)


    def __len__(self):
        image = self.dataset[0][0]
        w, h = image.size[0], image.size[1]
        num_patches_per_image = int(h * w / (self.patch_size**2))
        return len(self.dataset) * num_patches_per_image


if __name__ == "__main__":
    test_path="../data/liver2_mini/train"

    # PATCH DATASET
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




    # TRIPLET PATCH DATASET
    tripelt_dataset=datasets.TripletPatchDataset(root=test_path)

    triplet_sampler = TripletPatchSampler(dataset=tripelt_dataset, patch_size=16)

    transform = transforms.Compose([transforms.ToTensor()])
    tripelt_collate_fn = datasets.TripletBaseCollateFunction(transform)

    # TODO: figure out how to combine sampler with shuffle, like if shufle is True, pick a random image to start with?
    triplet_dataloader =  torch.utils.data.DataLoader(
        tripelt_dataset,
        sampler=triplet_sampler,
        batch_size=1,
        collate_fn=tripelt_collate_fn,
        shuffle=False,
        drop_last=True,
        num_workers=1,
    )

    print(f'Dataset size:{len(tripelt_dataset)}')
    print(f'Sampler size:{len(triplet_sampler)}')
    print(f'Dataloader size:{len(triplet_dataloader)}')

    for batch in triplet_dataloader:
        (anchor, _, _), (pos, _, _,), (neg, _, _) = batch
        print("anchor=", anchor.shape, ", pos=", pos.shape, ", neg=", neg.shape)
        break



