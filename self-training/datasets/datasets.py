from lightly.data import LightlyDataset, BaseCollateFunction
from PIL import Image
import numpy as np
import torchvision.transforms as T
from typing import List, Tuple
import torch

class TripletDataset(LightlyDataset):
    def __init__(
        self,
        root: str,
        transform: object = None,
    ):
        super(TripletDataset, self).__init__(root, transform)
        

    def __getitem__(self, index):
        a_index, p_index, n_index = self.get_random_triplet(index)
        
        # get filenames
        a_fname = self.index_to_filename(self.dataset, a_index)
        p_fname = self.index_to_filename(self.dataset, p_index)
        n_fname = self.index_to_filename(self.dataset, n_index)

        # get samples (image) and targets (label, eg foldername)
        a_sample, a_target = self.dataset.__getitem__(a_index)
        p_sample, p_target = self.dataset.__getitem__(p_index)
        n_sample, n_target = self.dataset.__getitem__(n_index)
        
        # Return the triplet of images
        return ((a_sample, a_target, a_fname), (p_sample, p_target, p_fname), (n_sample, n_target, n_fname))
    
    # TODO: add other approaches for striplet sampling
    def get_random_triplet(self, index):
        """
        Returns a triplet. Anchor, pos, neg are not the same.
        Returns:
            triplet (tuple of str): A tuple of 3 indices randomly selected from the dataset indices. 
        """
        anchor_index = index
        positive_index = np.random.choice(self.__len__())
        while positive_index==anchor_index:
            positive_index = np.random.choice(self.__len__())

        negative_index = np.random.choice(self.__len__())
        while negative_index==anchor_index or negative_index==positive_index:
            negative_index = np.random.choice(self.__len__())

        return (anchor_index, positive_index, negative_index)    

class TripletBaseCollateFunction(BaseCollateFunction):
    def __init__(self, transform: T.Compose):
        super(TripletBaseCollateFunction, self).__init__(transform)
        self.transform = transform

    def forward(self, batch: List[Tuple[ \
            Tuple[Image.Image, int, str], \
            Tuple[Image.Image, int, str], \
            Tuple[Image.Image, int, str]]]) \
                            -> Tuple[ \
            Tuple[torch.Tensor, torch.Tensor,torch.Tensor], \
            Tuple[torch.Tensor, torch.Tensor,torch.Tensor], \
            Tuple[torch.Tensor, torch.Tensor,torch.Tensor]]:
        """Turns a batch of triplet tuples into a tuple of batches.
        Args:
            batch:
                A batch of 3 tuples, each of tuple of images, labels, and filenames.
        Returns:
            A tuple of (anchors, labels, and filenames), (positives, labels, and filenames), (negatives, labels, and filenames)).
            The images consist of batches corresponding to transformations of the input images.
        Reference to basic collate function: https://github.com/lightly-ai/lightly/blob/master/lightly/data/collate.py
        """

        # lists of samples
        # anchors is 0th item in a tuple (a,p,n), anchor sample is 0th item in a tuple (sample, target, fname)
        a_samples = torch.stack([self.transform(item[0][0]) for item in batch])
        p_samples = torch.stack([self.transform(item[1][0]) for item in batch])
        n_samples = torch.stack([self.transform(item[2][0]) for item in batch])

        # lists of labels (targets)
        a_targets = torch.LongTensor([item[0][1] for item in batch])
        p_targets = torch.LongTensor([item[1][1] for item in batch])
        n_targets= torch.LongTensor([item[2][1] for item in batch])

        # lists of filenames
        a_fnames = [item[0][2] for item in batch]
        p_fnames = [item[1][2] for item in batch]
        n_fnames = [item[2][2] for item in batch]

        return (a_samples, a_targets, a_fnames), (p_samples, p_targets, p_fnames), (n_samples, n_targets, n_fnames)


if __name__ == "__main__":
    test_path="../data/liver2_mini_folders/train"

    dataset=TripletDataset(root=test_path)
    print(len(dataset))
    triplet = dataset[0]
    print("anchor=", triplet[0], ", pos=", triplet[1], ", neg=", triplet[2])