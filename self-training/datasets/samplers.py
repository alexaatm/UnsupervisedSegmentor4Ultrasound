import random
import torch
from torch.utils.data import Sampler
from torchvision.transforms import transforms
from datasets import datasets
from PIL import Image
import numpy as np

class PatchSampler(Sampler):
    def __init__(self, dataset, patch_size, patch_mode='random', remove_empty=True):
        self.dataset = dataset
        self.patch_size = patch_size
        self.patch_mode = patch_mode
        self.remove_empty = remove_empty

    def __iter__(self):
        indices = range(len(self.dataset))
        for idx in indices:
            image = self.dataset[idx][0]
            # TODO: check if image is C, W, H?
            h, w = image.size[-2], image.size[-1]
            if self.patch_mode == 'grid':
                # crop the image to a size that is evenly divisible by the patch size
                crop_height = h - h % self.patch_size
                crop_width = w - w % self.patch_size
                # TODO: check if you can use slicing here: im=im[:, :crop_height, : crop_width]?
                # check if PIl image crop is correct?
                image = image.crop((0, 0, crop_height, crop_width))
                for i in range(0, h - self.patch_size + 1, self.patch_size):
                    for j in range(0, w - self.patch_size + 1, self.patch_size):
                        patch = (idx, i, j, self.patch_size)
                        if self.remove_empty and np.sum(np.array(image)[..., i:i+self.patch_size, j:j+self.patch_size]) < (self.patch_size**2)*0.2:
                            continue
                        yield patch
            elif self.patch_mode == 'random':
                num_patches = int(h * w / (self.patch_size**2))
                for _ in range(num_patches):
                    i = torch.randint(high=h - self.patch_size + 1, size=(1,)).item()
                    j = torch.randint(high=w - self.patch_size + 1, size=(1,)).item()
                    patch = (idx, i, j, self.patch_size)
                    # TODO: check how to better check if patch is balck
                    # eg:
                    # if (np.count_nonzero(image[..., i:i+self.patch_size, j:j+self.patch_size]) / (self.patch_size*self.patch_size) < 0.2
                    if self.remove_empty and np.sum(np.array(image)[..., i:i+self.patch_size, j:j+self.patch_size]) < (self.patch_size**2)*0.2:
                        continue
                    yield patch

    def __len__(self):
        # TODO: check if this counting is reasonable - esp given you want to ignore the black patches
        # TODO: esp check if PIl size is C, H, W or smth different?
        if self.patch_mode == 'grid':
            return len(self.dataset) * ((self.dataset[0][0].size[-2] - self.patch_size + 1) // self.patch_size)**2
        elif self.patch_mode == 'random':
            return len(self.dataset) * (self.dataset[0][0].size[-2] * self.dataset[0][0].size[-1] // (self.patch_size**2))

if __name__ == "__main__":
    test_path="../data/liver_reduced/train"

    dataset=datasets.PatchDataset(root=test_path)

    sampler = PatchSampler(dataset=dataset, patch_size=16)

    transform = transforms.Compose([transforms.ToTensor()])
    collate_fn = datasets.BaseCollateFunction(transform)

    # TODO: figure out how to combine sampler with shuffle, like if shufle is True, pick a random image to start with?
    dataloader =  torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=10,
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



