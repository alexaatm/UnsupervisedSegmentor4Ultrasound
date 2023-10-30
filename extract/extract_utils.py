import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import cv2
import numpy as np
import scipy.sparse
import torch
from skimage.morphology import binary_dilation, binary_erosion
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

import random
import os
import pytorch_lightning as pl


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


def get_model(name: str):
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

def get_model_from_checkpoint(model_name: str, ckpt_path: str, just_backbone=False):
    if 'dino' in model_name:
        if just_backbone:
            # get the backbone
            model = torch.hub.load('facebookresearch/dino:main', model_name, pretrained=False)
            input_dim = model.embed_dim

            # load the backbone model from the checkpoint
            device='cuda' if torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
            print(checkpoint.keys())
            state_dict = checkpoint['state_dict']

             # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            # use strict=False to ignore projection head layers..
            model.load_state_dict(state_dict, strict=False)
            model.fc = torch.nn.Identity()
            num_heads = model.blocks[0].attn.num_heads
            patch_size = model.patch_embed.patch_size

            val_transform = get_transform(model_name)
    else:
        raise ValueError(f'Cannot get model: {model_name}')
    model = model.eval()
    return model, val_transform, patch_size, num_heads

def get_transform(name: str):
    if any(x in name for x in ('dino', 'mocov3', 'convnext', )):
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transform = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        raise NotImplementedError()
    return transform


def get_inverse_transform(name: str):
    if 'dino' in name:
        inv_normalize = transforms.Normalize(
            [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            [1 / 0.229, 1 / 0.224, 1 / 0.225])
        transform = transforms.Compose([transforms.ToTensor(), inv_normalize])
    else:
        raise NotImplementedError()
    return transform


def get_image_sizes(data_dict: dict, downsample_factor: Optional[int] = None):
    P = data_dict['patch_size'] if downsample_factor is None else downsample_factor
    B, C, H, W = data_dict['shape']
    assert B == 1, 'assumption violated :('
    H_patch, W_patch = H // P, W // P
    H_pad, W_pad = H_patch * P, W_patch * P
    return (B, C, H, W, P, H_patch, W_patch, H_pad, W_pad)


def _get_files(p: str):
    if Path(p).is_dir():
        return sorted(Path(p).iterdir())
    elif Path(p).is_file():
        return Path(p).read_text().splitlines()
    else:
        raise ValueError(p)


def get_paired_input_files(path1: str, path2: str):
    files1 = _get_files(path1)
    files2 = _get_files(path2)
    assert len(files1) == len(files2)
    return list(enumerate(zip(files1, files2)))


def make_output_dir(output_dir, check_if_empty=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    if check_if_empty and (len(list(output_dir.iterdir())) > 0):
        print(f'Output dir: {str(output_dir)}')
        if input(f'Output dir already contains files. Continue? (y/n) >> ') != 'y':
            sys.exit()  # skip because already generated


def get_largest_cc(mask: np.array):
    from skimage.measure import label as measure_label
    labels = measure_label(mask)  # get connected components
    largest_cc_index = np.argmax(np.bincount(labels.flat)[1:]) + 1
    largest_cc_mask = (labels == largest_cc_index)
    return largest_cc_mask


def erode_or_dilate_mask(x: Union[torch.Tensor, np.ndarray], r: int = 0, erode=True):
    fn = binary_erosion if erode else binary_dilation
    for _ in range(r):
        x_new = fn(x)
        if x_new.sum() > 0:  # do not erode the entire mask away
            x = x_new
    return x


def get_border_fraction(segmap: np.array):
    num_border_pixels = 2 * (segmap.shape[0] + segmap.shape[1])
    counts_map = {idx: 0 for idx in np.unique(segmap)}
    np.zeros(len(np.unique(segmap)))
    for border in [segmap[:, 0], segmap[:, -1], segmap[0, :], segmap[-1, :]]:
        unique, counts = np.unique(border, return_counts=True)
        for idx, count in zip(unique.tolist(), counts.tolist()):
            counts_map[idx] += count
    # normlized_counts_map = {idx: count / num_border_pixels for idx, count in counts_map.items()}
    indices = np.array(list(counts_map.keys()))
    normlized_counts = np.array(list(counts_map.values())) / num_border_pixels
    return indices, normlized_counts


def parallel_process(inputs: Iterable, fn: Callable, multiprocessing: int = 0):
    start = time.time()
    if multiprocessing:
        print('Starting multiprocessing')
        with Pool(multiprocessing) as pool:
            for _ in tqdm(pool.imap(fn, inputs), total=len(inputs)):
                pass
    else:
        for inp in tqdm(inputs):
            fn(inp)
    print(f'Finished in {time.time() - start:.1f}s')


def knn_affinity(image, n_neighbors=[20, 10], distance_weights=[2.0, 0.1]):
    """Computes a KNN-based affinity matrix. Note that this function requires pymatting"""
    try:
        from pymatting.util.kdtree import knn
    except:
        raise ImportError(
            'Please install pymatting to compute KNN affinity matrices:\n'
            'pip3 install pymatting'
        )

    h, w = image.shape[:2]
    r, g, b = image.reshape(-1, 3).T
    n = w * h

    x = np.tile(np.linspace(0, 1, w), h)
    y = np.repeat(np.linspace(0, 1, h), w)

    i, j = [], []

    for k, distance_weight in zip(n_neighbors, distance_weights):
        f = np.stack(
            [r, g, b, distance_weight * x, distance_weight * y],
            axis=1,
            out=np.zeros((n, 5), dtype=np.float32),
        )

        distances, neighbors = knn(f, f, k=k)

        i.append(np.repeat(np.arange(n), k))
        j.append(neighbors.flatten())

    ij = np.concatenate(i + j)
    ji = np.concatenate(j + i)
    coo_data = np.ones(2 * sum(n_neighbors) * n)

    # This is our affinity matrix
    W = scipy.sparse.csr_matrix((coo_data, (ij, ji)), (n, n))
    return W


def rw_affinity(image, sigma=0.033, radius=1):
    """Computes a random walk-based affinity matrix. Note that this function requires pymatting"""
    try:
        from pymatting.laplacian.rw_laplacian import _rw_laplacian
    except:
        raise ImportError(
            'Please install pymatting to compute RW affinity matrices:\n'
            'pip3 install pymatting'
        )
    h, w = image.shape[:2]
    n = h * w
    values, i_inds, j_inds = _rw_laplacian(image, sigma, radius)
    W = scipy.sparse.csr_matrix((values, (i_inds, j_inds)), shape=(n, n))
    return W


def get_diagonal(W: scipy.sparse.csr_matrix, threshold: float = 1e-12):
    """Gets the diagonal sum of a sparse matrix"""
    try:
        from pymatting.util.util import row_sum
    except:
        raise ImportError(
            'Please install pymatting to compute the diagonal sums:\n'
            'pip3 install pymatting'
        )

    D = row_sum(W)
    D[D < threshold] = 1.0  # Prevent division by zero.
    D = scipy.sparse.diags(D)
    return D

def reshape_split(image: np.ndarray, kernel_size: tuple):
  """
  Computes non-overlapping patches for a given image and a given patch size.
  Note that th eimage should be able to fit a whole number of patches of the given size.
  # based on: https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
  """
  h, w, ch = image.shape
  tile_h, tile_w = kernel_size

  tiled_array=image.reshape(h//tile_h, 
                            tile_h, 
                            w//tile_w,
                            tile_w,
                            ch)
  tiled_array=tiled_array.swapaxes(1,2)
  tiled_array=tiled_array.reshape(-1,tile_h,tile_w,ch)
  return tiled_array

def ssd_patchwise_affinity_knn(image, patch_size, n_neighbors=[8, 4], distance_weights=[2.0, 0.1]):
  """
  Computes a SSD-based affinity matrix for patches of a single image.
  Note that this function requires pymattin and scipy.

  step 1 - split image into patches
  step 2 - flatten patches along x,y and ch dimensions -> results in shape (num_patches, rest)
  step 3 - calculate position arrays for distance weighing
  step 4 - apply knn approach, concatenating flattened patches with weighted position arrays (different for diff distance weights)
  step 5 - assemble affinity matrix

  par: image - ndarray, of size compatible with the patch size, normalized
  par: patch_size - a tuple (patch_height, patch_width)
  
  based on: https://github.com/pymatting/pymatting/blob/master/pymatting/laplacian/knn_laplacian.py
  """
  try:
    from pymatting.util.kdtree import knn
  except:
        raise ImportError(
            'Please install pymatting to compute KNN affinity matrices:\n'
            'pip3 install pymatting'
        )
  
  patches=reshape_split(image, patch_size)
  
  patches_2d = patches.reshape(patches.shape[0],-1)

  n_patches=patches_2d.shape[0]
  n_height=image.shape[0]//patch_size[0]
  n_width=image.shape[1]//patch_size[1]
  x = np.tile(np.linspace(0, 1, n_width), n_height)
  y = np.repeat(np.linspace(0, 1, n_height), n_width)

  i, j = [], []

  for k, distance_weight in zip(n_neighbors, distance_weights):
    xs=(distance_weight * x)[:, None]
    ys=(distance_weight * y)[:, None]
    f = np.concatenate((patches_2d, xs, ys), axis = 1, dtype=np.float32)
    distances, neighbors = knn(f, f, k=k)
    i.append(np.repeat(np.arange(n_patches), k))
    j.append(neighbors.flatten())

  ij = np.concatenate(i + j)
  ji = np.concatenate(j + i)
  coo_data = np.ones(2 * sum(n_neighbors) * n_patches)

  # This is our affinity matrix
  W = scipy.sparse.csr_matrix((coo_data, (ij, ji)), (n_patches, n_patches)) 

  # Convert to dense numpy array
  W = np.array(W.todense().astype(np.float32))

  return W, patches

from PIL import Image

def interpolate_2Darray(input_2Darray, output_size):
  """
  based on : PIL Image functionality for interpolating images when resizing
  """
  image_from_array = Image.fromarray(input_2Darray).resize((output_size[0],output_size[1]), Image.BILINEAR)
  array_from_image = np.array(image_from_array)
  return array_from_image


def var_patchwise_affinity_knn(image, patch_size, n_neighbors=[8, 4], distance_weights=[0.0, 0.0]):
  """
    UPDATE: use distance_weigts of [0, 0] to avoid position of patches overpowering the variance values... 
  Computes a SSD-based affinity matrix for VARIANCE of patches of a single image.
  Note that this function requires pymattin and scipy.

  step 1 - split image into patches
  step 2 - calculate variance of each patch
  step 3 - calculate position arrays for distance weighing
  step 4 - apply knn approach, concatenating patch variances with weighted position arrays (different for diff distance weights)
  step 5 - assemble affinity matrix

  par: image - ndarray, of size compatible with the patch size, normalized
  par: patch_size - a tuple (patch_height, patch_width)
  
  based on: https://github.com/pymatting/pymatting/blob/master/pymatting/laplacian/knn_laplacian.py
  """
  try:
    from pymatting.util.kdtree import knn
  except:
        raise ImportError(
            'Please install pymatting to compute KNN affinity matrices:\n'
            'pip3 install pymatting'
        )
  
  patches=reshape_split(image, patch_size)
  
#   patches_2d = patches.reshape(patches.shape[0],-1)

#   n_patches=patches_2d.shape[0]
  n_patches = len(patches)
  n_height=image.shape[0]//patch_size[0]
  n_width=image.shape[1]//patch_size[1]

  var_patchwise = []  
  for p in patches:
    var_patchwise.append([np.var(p)])
  x = np.tile(np.linspace(0, 1, n_width), n_height)
  y = np.repeat(np.linspace(0, 1, n_height), n_width)

  i, j = [], []

  for k, distance_weight in zip(n_neighbors, distance_weights):
    xs=(distance_weight * x)[:, None]
    ys=(distance_weight * y)[:, None]
    f = np.concatenate((var_patchwise, xs, ys), axis = 1, dtype=np.float32)
    distances, neighbors = knn(f, f, k=k)
    i.append(np.repeat(np.arange(n_patches), k))
    j.append(neighbors.flatten())

  ij = np.concatenate(i + j)
  ji = np.concatenate(j + i)
  coo_data = np.ones(2 * sum(n_neighbors) * n_patches)

  # This is our affinity matrix
  W = scipy.sparse.csr_matrix((coo_data, (ij, ji)), (n_patches, n_patches)) 

  # Convert to dense numpy array
  W = np.array(W.todense().astype(np.float32))

  return W, patches

def set_seed(seed: int = 1) -> None:
    # ref: https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # additionally, ust in case, do
    pl.seed_everything(1)
    print(f"Random seed set as {seed}")
