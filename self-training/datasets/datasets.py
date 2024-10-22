from lightly.data import LightlyDataset, BaseCollateFunction
from PIL import Image
import numpy as np
import torchvision.transforms as T
from typing import List, Tuple
import torch
import cv2
from datasets import dataset_utils
from datasets import samplers

class TripletDataset(LightlyDataset):
    def __init__(
        self,
        root: str,
        transform: object = None,
        mode: str ='seq',
    ):
        super(TripletDataset, self).__init__(root, transform)
        self.mode=mode
        if self.mode == 'seq':
            self.relabeled_list, self.classes_list = dataset_utils.detect_shots_from_list_label(self.dataset)
            self.set_dataset(self.relabeled_list)
        elif self.mode == 'class':
            self.classes_list = dataset_utils.get_unique_classes_list(self.dataset)
        print(f"Unique class labels found: {self.classes_list}")
        

    def __getitem__(self, index):
        if self.mode == 'random':
            a_index, p_index, n_index = self.get_random_triplet(index)
        elif self.mode == 'seq':
            a_index, p_index, n_index = self.get_triplet_by_class(self.relabeled_list, self.classes_list)
        elif self.mode == 'class':
            a_index, p_index, n_index = self.get_triplet_by_class(self.dataset, self.classes_list)
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

    def get_triplet_by_class(self, labeled_list, classes_list):
        """
        Returns a triplet. Anchor, pos, neg are not the same based
        on a class given by analizing changes in frame sequence.
        Returns:
            triplet (tuple of str): A tuple of 3 indices selected from the dataset classes.
        
        Ref. for sampling based on classes: https://github.com/andreasveit/triplet-network-pytorch/blob/master/triplet_mnist_loader.py
        """
        image_labels_np = np.array([label for _, label in labeled_list])

        # pick a class randomly
        class_idx = np.random.choice(classes_list)
        # print(f'class : {class_idx}')
        anchor_index = np.random.choice(np.where(image_labels_np==class_idx)[0])
        positive_index = np.random.choice(np.where(image_labels_np==class_idx)[0])
        if (len(np.where(image_labels_np==class_idx)[0])>1):
            # print(f'class {class_idx} has {np.where(image_labels_np==class_idx)[0]} samples')
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
    def __init__(self, transform: T.Compose, pos_transform: T.Compose):
        super(TripletBaseCollateFunction, self).__init__(transform)
        self.transform = transform
        if pos_transform==None:
            self.pos_transform=T.Compose([])
        else:
            self.pos_transform = pos_transform

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
        p_samples = torch.stack([self.pos_transform(self.transform(item[1][0])) for item in batch])
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

class PatchDataset(LightlyDataset):
    def __init__(
        self,
        root: str,
        transform: object = None,
    ):
        super(PatchDataset, self).__init__(root, transform)        

    def __getitem__(self, index):
        # print(f'index={index}')
        if isinstance(index, tuple):
            idx, i, j, patch_size = index
            # get filename
            fname = self.index_to_filename(self.dataset, idx)

            # get samples (image) and targets (label)
            sample, target = self.dataset.__getitem__(idx)

            # get a specified patch
            # patch = sample[..., i:i+patch_size, j:j+patch_size]
            # TODO: check if you need to switch H and W for PIL Image -> i for width, j for height, PIL image has (W, H)
            patch = sample.crop((i, j, i+patch_size, j+patch_size))
            # patch.show()

            return (patch, target, f'patch_{i}_{j}_{fname}')

        else:
            # just return a full image
            # get filename
            fname = self.index_to_filename(self.dataset, index)

            # get samples (image) and targets (label)
            sample, target = self.dataset.__getitem__(index)

            return (sample, target, fname)

class TripletPatchDataset(LightlyDataset):
    def __init__(
        self,
        root: str,
        transform: object = None,
    ):
        super(TripletPatchDataset, self).__init__(root, transform)    

    def __getitem__(self, index):
        # print(f'index={index}')
        if isinstance(index, tuple):
            idx, a, p, n, patch_size = index
            # get filename
            fname = self.index_to_filename(self.dataset, idx)

            # get samples (image) and targets (label)
            sample, target = self.dataset.__getitem__(idx)

            # get specified patches for a triplet
            a_sample = sample.crop((a[0], a[1], a[0]+patch_size, a[1]+patch_size))
            p_sample = sample.crop((p[0], p[1], p[0]+patch_size, p[1]+patch_size))
            n_sample = sample.crop((n[0], n[1], n[0]+patch_size, n[1]+patch_size))

            # patch.show()

            return ((a_sample, target, f'patch_{a[0]}_{a[1]}_{fname}'), \
                    (p_sample, target, f'patch_{p[0]}_{p[1]}_{fname}'), \
                    (n_sample, target, f'patch_{n[0]}_{n[1]}_{fname}'))
        else:
            # just return a full image
            # get filename
            fname = self.index_to_filename(self.dataset, index)

            # get samples (image) and targets (label)
            sample, target = self.dataset.__getitem__(index)

            return (sample, target, fname)


if __name__ == "__main__":
    # test_path="../data/liver_reduced/train"
    # test_path="../data/liver_similar"
    test_path="../data/imagenet-4-classes/train"


    dataset=TripletDataset(root=test_path, mode='class')
    print(len(dataset))
    triplet = dataset[0]
    print("anchor=", triplet[0], ", pos=", triplet[1], ", neg=", triplet[2])

    images = dataset.get_dataset()
    print(len(images))
    sample = images[0]
    print(f'Sample= {sample}')


    


