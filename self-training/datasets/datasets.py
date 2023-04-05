from lightly.data import LightlyDataset
from typing import List, Tuple
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as T

class TripletDataset(LightlyDataset):
    def __init__(
        self,
        root: str,
        transform: object = T.ToTensor(),
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
        return [(a_sample, a_target, a_fname), (p_sample, p_target, p_fname), (n_sample, n_target, n_fname)]
    
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


if __name__ == "__main__":
    test_path="../data/liver2_mini_folders/train"

    dataset=TripletDataset(root=test_path)
    print(len(dataset))
    triplet = dataset[0]
    print("anchor=", triplet[0], ", pos=", triplet[1], ", neg=", triplet[2])