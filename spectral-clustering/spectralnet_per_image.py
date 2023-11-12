import torch
import numpy as np
from spectralnet import SpectralNet
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
from PIL import Image


def get_image_sizes(data_dict: dict, downsample_factor = None):
    P = data_dict['patch_size'] if downsample_factor is None else downsample_factor
    B, C, H, W = data_dict['shape']
    assert B == 1, 'assumption violated :('
    H_patch, W_patch = H // P, W // P
    H_pad, W_pad = H_patch * P, W_patch * P
    return (B, C, H, W, P, H_patch, W_patch, H_pad, W_pad)

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

def get_segmap(clusters, data_dict):
    B, C, H, W, P, H_patch, W_patch, H_pad, W_pad = get_image_sizes(data_dict)
    # Reshape
    infer_bg_index = True
    if clusters.size == H_patch * W_patch:  # TODO: better solution might be to pass in patch index
        segmap = clusters.reshape(H_patch, W_patch)
    elif clusters.size == H_patch * W_patch * 4:
        segmap = clusters.reshape(H_patch * 2, W_patch * 2)
    else:
        raise ValueError()

    # TODO: Improve this step in the pipeline.
    # Background detection: we assume that the segment with the most border pixels is the 
    # background region. We will always make this region equal 0. 
    if infer_bg_index:
        indices, normlized_counts = get_border_fraction(segmap)
        bg_index = indices[np.argmax(normlized_counts)].item()
        bg_region = (segmap == bg_index)
        zero_region = (segmap == 0)
        segmap[bg_region] = 0
        segmap[zero_region] = bg_index

    return segmap

feature_path = "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/deep-spectral-segmentation/outputs/liver_mixed_val_mini/exp_clustering_sweep/2023-11-07/13-13-13/seg8_clust6_norm-imagenet_prepr-False_dino1_clusterkmeans/features/dino_vits8/Patient-12-david-01_7.pth"
eigs_path = "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/deep-spectral-segmentation/outputs/liver_mixed_val_mini/exp_clustering_sweep/2023-11-07/13-13-13/seg8_clust6_norm-imagenet_prepr-False_dino1_clusterkmeans/eig/laplacian/Patient-12-david-01_7.pth"
image_path = "/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/LIVER_MIXED/val_mini/images/Patient-12-david-01_7.png"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
data_dict = torch.load(feature_path, map_location=device) # map_location='cpu'
data_dict.update(torch.load(eigs_path), map_location=device)
    
feats = data_dict['k'].squeeze()
feats2 = data_dict['k'].squeeze()
feats_numpy = feats.cpu().detach().numpy()
n_clusters = 6

print(f'feats are on device: {feats.device}')
print(f'feats2 are on device: {feats2.device}')


# kmeans baseline
kmeans = KMeans(n_clusters=n_clusters, random_state=1)
clusters1 = kmeans.fit_predict(feats_numpy)

# eigenvectors (deep spectral approach)
# num_eigenvectors = 1000000
# eigenvectors = data_dict['eigenvectors'][1:1+num_eigenvectors].numpy()  # take non-constant eigenvectors
# clusters2 = kmeans.fit_predict(eigenvectors.T)

# spectral_net
spectralnet = SpectralNet(n_clusters=n_clusters,
                        should_use_siamese=True,
                        should_use_ae = True)
spectralnet.fit(feats)
clusters3 = spectralnet.predict(feats)

print(f'feats are on device: {feats.device}')
print(f'feats2 are on device: {feats2.device}')

spectralnet2 = SpectralNet(n_clusters=n_clusters,
                        should_use_siamese=True,
                        should_use_ae = True,
                        is_sparse_graph=True,
                        spectral_n_nbg=2)

feats2 = feats2.to('cuda')
print(f'feats are on device: {feats.device}')
print(f'feats2 are on device: {feats2.device}')
spectralnet2.fit(feats2)
clusters4 = spectralnet2.predict(feats2)

# get segmentation maps
segmap_kmeans = get_segmap(clusters1, data_dict)
# segmap_eigen = get_segmap(clusters2, data_dict)
segmap_spectralnet = get_segmap(clusters3, data_dict)
segmap_spectralnet2 = get_segmap(clusters4, data_dict)


# create a plot
image = np.array(Image.open(image_path))

fig, axs = plt.subplots(1, 4, figsize=(8, 8))

axs[0].imshow(image)
axs[0].set_title("image")

axs[1].imshow(segmap_kmeans)
axs[1].set_title("dino_kmeans")

axs[2].imshow(segmap_spectralnet)
axs[2].set_title("spectralnet_nonsparse_nn30")

axs[3].imshow(segmap_spectralnet2)
axs[3].set_title("spectralnet_sparse_nn2")

plt.tight_layout()
fig.savefig("/home/guests/oleksandra_tmenova/test/project/thesis-codebase/spectral-clustering/per_image_clustering_6segments.png")



