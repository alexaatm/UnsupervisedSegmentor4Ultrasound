from lightly.data import LightlyDataset
from typing import List, Tuple
from PIL import Image
import torch
import random
from utils import utils

class TripletDataset(LightlyDataset):
    def __init__(
        self,
        root: str,
        transform: object = None,
        extensions: List[str] = None,
        n_triplets: int = 1
    ):
        super(TripletDataset, self).__init__(root, transform)
        self.n_triplets = n_triplets
        self.image_paths = utils.get_file_paths_recursive(root, extensions)

    def __getitem__(self, index):
        anchor, positive, negative = self.get_random_triplet()
        
        # Load the images
        anchor_image = Image.open(anchor).convert('RGB')
        positive_image = Image.open(positive).convert('RGB')
        negative_image = Image.open(negative).convert('RGB')
        
        # Apply the transformations
        if self.transform is not None:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)
        
        # Return the triplet of images
        return anchor_image, positive_image, negative_image
    
    def get_random_triplet(self):
        """
        Returns a triplet of 3 image paths randomly from a list of image paths. Anchor, pos, neg are not the same.

        Parameters:
            image_paths (list of str): List of image paths.

        Returns:
            triplet (tuple of str): A tuple of 3 image paths randomly selected from the input list. 
        """
        anchor_path = random.choice(self.image_paths)

        positive_paths = [path for path in self.image_paths if path != anchor_path]
        positive_path = random.choice(positive_paths)

        negative_paths = [path for path in self.image_paths if path != anchor_path and path != positive_path]
        negative_path = random.choice(negative_paths)

        return (anchor_path, positive_path, negative_path)
    
    def collate_fn(self, samples: List[Tuple[Image.Image, Image.Image, Image.Image]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors = torch.stack([s[0] for s in samples])
        positives = torch.stack([s[1] for s in samples])
        negatives = torch.stack([s[2] for s in samples])

        return anchors, positives, negatives
    


if __name__ == "__main__":
    test_path="../data/liver2_mini/train"

    dataset=TripletDataset(root=test_path)
    triplet = dataset[0]
    print("anchor=", triplet[0], ", pos=", triplet[1], ", neg=", triplet[2])