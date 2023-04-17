from lightly.data import LightlyDataset, BaseCollateFunction
from PIL import Image
import numpy as np
import torchvision.transforms as T
from typing import List, Tuple
import torch
import cv2
from datasets import dataset_utils

class TripletDataset(LightlyDataset):
    def __init__(
        self,
        root: str,
        transform: object = None,
        mode: str ='seq',
    ):
        super(TripletDataset, self).__init__(root, transform)
        self.mode=mode
        

    def __getitem__(self, index):
        if self.mode == 'random':
            a_index, p_index, n_index = self.get_random_triplet(index)
        elif self.mode == 'seq':
            relabeled_list, classes_list = dataset_utils.detect_shots_from_list_label(self.dataset)
            self.set_dataset(relabeled_list)
            print(f"Unique class labels found: {classes_list}")
            a_index, p_index, n_index = self.get_triplet_by_seq_class(relabeled_list, classes_list)
        else:
            print(f"No sampling mode called {self.mode}")
            raise NotImplementedError()
        
        # get filenames
        a_fname = self.index_to_filename(self.dataset, a_index)
        p_fname = self.index_to_filename(self.dataset, p_index)
        n_fname = self.index_to_filename(self.dataset, n_index)

        # get samples (image) and targets (label)
        a_sample, a_target = self.dataset.__getitem__(a_index)
        p_sample, p_target = self.dataset.__getitem__(p_index)
        n_sample, n_target = self.dataset.__getitem__(n_index)
        
        # Return the triplet of images
        return ((a_sample, a_target, a_fname), (p_sample, p_target, p_fname), (n_sample, n_target, n_fname))
    
    # TODO: add other approaches for triplet sampling
    def get_random_triplet(self, index):
        """
        Returns a triplet. Anchor, pos, neg are not the same.
        Returns:
            triplet (tuple of str): A tuple of 3 indices randomly selected from the dataset indices. 
        """
        # TODO: choose anchor index also randomly from the whole dataset
        anchor_index = index
        positive_index = np.random.choice(self.__len__())
        while positive_index==anchor_index:
            positive_index = np.random.choice(self.__len__())

        negative_index = np.random.choice(self.__len__())
        while negative_index==anchor_index or negative_index==positive_index:
            negative_index = np.random.choice(self.__len__())

        return (anchor_index, positive_index, negative_index)    

    def get_triplet_by_seq_class(self, relabeled_list, classes_list):
        """
        Returns a triplet. Anchor, pos, neg are not the same based
        on a class given by analizing changes in frame sequence.
        Returns:
            triplet (tuple of str): A tuple of 3 indices selected from the dataset classes.
        
        Ref. for sampling based on classes: https://github.com/andreasveit/triplet-network-pytorch/blob/master/triplet_mnist_loader.py
        """
        image_labels_np = np.array([label for _, label in relabeled_list])
        for class_idx in classes_list:
            anchor_index = np.random.choice(np.where(image_labels_np==class_idx)[0])
            positive_index = np.random.choice(np.where(image_labels_np==class_idx)[0])
            while positive_index==anchor_index:
                positive_index = np.random.choice(np.where(image_labels_np==class_idx)[0])
            negative_index = np.random.choice(np.where(image_labels_np!=class_idx)[0])
        return (anchor_index, positive_index, negative_index)

    def get_dataset(self):
        return self.dataset
    
    def set_dataset(self, ims_labels):
        # TODO: check if corresponds to the needed format: list of tuples (PIL IMAGE, int label)
        self.dataset = ims_labels
    
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
    # test_path="../data/liver_reduced/train"
    test_path="../data/liver_similar"

    dataset=TripletDataset(root=test_path, mode='seq')
    print(len(dataset))
    triplet = dataset[0]
    print("anchor=", triplet[0], ", pos=", triplet[1], ", neg=", triplet[2])

    images = dataset.get_dataset()
    print(len(images))
    sample = images[0]
    print(f'Sample= {sample}')

    # opencvImage = cv2.cvtColor(np.array(sample[0]), cv2.COLOR_RGB2BGR)
    # print(f'Sample= {opencvImage}')

    # pil_images = [im[0] for im in images]
    # changes = dataset_utils.detect_shots_from_list(pil_images)

    # changed_list, _ = dataset_utils.detect_shots_from_list_label(images)
    # dataset.set_dataset(changed_list)
    # print(len(dataset))
    # triplet = dataset[0]
    # print("anchor=", triplet[0], ", pos=", triplet[1], ", neg=", triplet[2])

