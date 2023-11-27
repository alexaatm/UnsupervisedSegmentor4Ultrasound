from pathlib import Path
from typing import Callable, Optional, Tuple, Any

import cv2
import torch
from skimage.morphology import binary_dilation, binary_erosion
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import argparse
import os

class ImagesDataset(Dataset):
    """A very simple dataset for loading images."""

    def __init__(self, filenames: str, images_root: Optional[str] = None, transform: Optional[Callable] = None,
                 prepare_filenames: bool = True) -> None:
        self.root = None if images_root is None else Path(images_root)
        self.filenames = sorted(list(set(filenames))) if prepare_filenames else filenames
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = self.filenames[index]
        full_path = Path(path) if self.root is None else self.root / path
        assert full_path.is_file(), f'Not a file: {full_path}'
        image = cv2.imread(str(full_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image, path, index

    def __len__(self) -> int:
        return len(self.filenames)
    
class OnlineMeanStd:
    """ A class for calculating mean and std of a given dataset
        ref: https://github.com/Nikronic/CoarseNet/blob/master/utils/preprocess.py#L142-L200
    """
    def __init__(self):
        pass

    def __call__(self, dataset, batch_size, method='strong'):
        """
        Calculate mean and std of a dataset in lazy mode (online)
        On mode strong, batch size will be discarded because we use batch_size=1 to minimize leaps.

        :param dataset: Dataset object corresponding to your dataset
        :param batch_size: higher size, more accurate approximation
        :param method: weak: fast but less accurate, strong: slow but very accurate - recommended = strong
        :return: A tuple of (mean, std) with size of (3,)
        """

        if method == 'weak':
            loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=0)
            mean = 0.
            std = 0.
            nb_samples = 0.
            for item in loader:
                data, files, indices = item
                batch_samples = data.size(0)
                data = data.view(batch_samples, data.size(1), -1)
                mean += data.mean(2).sum(0)
                std += data.std(2).sum(0)
                nb_samples += batch_samples

            mean /= nb_samples
            std /= nb_samples

            return mean, std

        elif method == 'strong':
            loader = DataLoader(dataset=dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=0)
            cnt = 0
            fst_moment = torch.empty(3)
            snd_moment = torch.empty(3)

            for item in loader:
                data, files, indices = item
                b, c, h, w = data.shape
                nb_pixels = b * h * w
                sum_ = torch.sum(data, dim=[0, 2, 3])
                sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
                fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
                snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

                cnt += nb_pixels

            return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
        

def find_custom_norm_mean_std(images_list, images_root, batch_size):
    filenames = Path(images_list).read_text().splitlines()
    dataset_raw = ImagesDataset(filenames=filenames, images_root=images_root, transform=transforms.ToTensor())
    meanStdCalculator = OnlineMeanStd()
    mean, std = meanStdCalculator(dataset_raw, batch_size=batch_size, method='strong')
    return mean, std

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find a custom mean and std of a given dataset')
    parser.add_argument('--images_list', type=str, 
                        help='Path to the txt file containing filenames of the dataset (for compatibility with deep spectral dataset format)')
    parser.add_argument('--images_root', type=str, 
                        help='Path to the images folder of the dataset')
    parser.add_argument('--batch_size', type=str, 
                        default=100,
                        help='A batchsize for finding mean and std of a datset in batches (the higher the better)')
    
    args = parser.parse_args()

    # train dataset for US mixed
    # images_root="/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/US_MIXED/train/images"
    # images_list="/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/US_MIXED/train/lists/images.txt"

    mean, std = find_custom_norm_mean_std(args.images_list, args.images_root, args.batch_size)
    print(f"Dataset_root = {args.images_root}, batch_size = {args.batch_size}")
    print(f"Mean = {mean}, std = {std}")


