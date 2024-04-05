
# Deep Spectral Methods for Unsupervised Ultrasound Image Interpretation 

We integrate key concepts from unsupervised deep spectral methods, which combine spectral graph theory with deep learning methods. We utilize self-supervised transformer features for spectral clustering to generate meaningful segments based on ultrasound-specific metrics and shape and positional priors, ensuring semantic consistency across the dataset.



## Installation

We recommend using a conda environment following steps

```bash
conda env create -f conda.yaml
conda activate dss
```
Alternatively, you can install the packages yourself: conda packages are listed in ``req_conda.txt`` file, and other packages not pressent in conda (pip) are listed in ``req_pip.txt``.

```bash
conda create --name dss
conda activate dss
conda install --name dss --yes --file req_conda.txt -c fastai -c defaults -c conda-forge
conda install --name dss --yes pip
<conda pip path> install -r req_pip.txt
```

This installation worked with Python 3.11.


## Organization

To run this on your data, you need to prepare the dataset itself (2D images), config for the dataset, and config for the run (the desired parameters).

### Data
    
Put data in the `data` folder of main repo. `lists` folder has a file `images.txt` with the list of files. `images` and `labels` are `.png` files with mathcing names. The folder should follow the structure:
    

```markdown
data/
--DATSET1
----subfolder1/
------images
------lists
------labels
...
----subfolderN/
------images
------lists
------labels
```

See `data/README.me` for more details.

### Data config

Add a dataset `.yaml` config to `configs/dataset`. It should follow the structure (example):

```markdown
name: carotid
dataset_root: carotid/val
dataset_type: folders
images_root: images
list: lists/images.txt
gt_dir: labels
pred_dir: ""
n_classes: 2
features_dir: ""
preprocessed_dir: ""
derained_dir: ""
eigenseg_dir: ""
```
See `configs/dataset/README.me` for another example.

### Run config

There are two ways to set parameters of the pipeline: through the hydra config when running the python script (see Next Section), or through a wandb sweep config (described here).

To use sweep config, add a new (or modify an existing) yaml file in the `configs/sweep` folder. It has few parameters of the sweep itself (name, type, count), and custom parameters (the config dictionary itself, the steps to evaluate - relevant only if evaluation is on). See [wandb sweeps](https://docs.wandb.ai/guides/sweeps) for more details. You should put here parameters you want to modify (otherwise they have theh value form their default configs). Here is a simple sweep config to explore multiple cluster numbers for the Step II of the pipeline (where eigensegments get merged into semantic clusters).


```markdown
name: num_clusters
seg_for_eval: ['crf_multi_region']
method: grid
count: 
simple: True
sweep_id: null
config:
    # generic
    segments_num: [15]
    clusters_num: [6,9,12,15]
```
See other examples of sweep configs in `configs/sweep` folder.

Note: Sweep config yaml file allows a more succinct way to set multiple values for parameters in order to run the code on multiple values and to let wandb take care of sweeps (such sweeps are nice for evaluation purposes and are easier to control through wanbd compared to hydra sweeps where each run is independant). If you just set one value, then it will be equivalent to a single run. 

It is nice to use swepe config files, because it makes it easier to track what configurations have been tried (as opposed to changing parameters through hydra configs when calling the python script).

### wandb config

You can set the name of the project and the wandb authentication key, to track the pipeline progress in wand. Especially good for evaluation, since all he results and metrics are also logged there. 
## Usage/Examples

In the root directory, there is a bash file with an example call for the pipeline.

```bash
cd deep-spectral-segmentation

export WANDB_API_KEY=<your-wandb-api>
export WANDB_CONFIG_DIR=/tmp/
export WANDB_CACHE_DIR=/tmp/
export WANDB_AGENT_MAX_INITIAL_FAILURE=20
export WANDB__SERVICE_WAIT=600
export XFORMERS_DISABLED=True

python -m pipeline.pipeline_sweep_subfolders \
    vis=selected \
    pipeline_steps=defaults \
    dataset=thyroid \
    wandb.tag=test \
    sweep=defaults
```

Configs are set through the hydra .yaml configs. For example: 
- `selected.yaml` config in `configs/vis` indicates which steps should be visualized
- `defaults.yaml` config in `configs/pipeline_steps` indicates that all steps, except for the evaluation should be completed. Other options include `allTrue` (to generate pseudomasks and to evaluate) or `allFalse` (e.g. when you use precomputed paths and only want to add plots).
- `thyroid.yaml` config in `configs/dataset` described which dataset to run the pipeline on.
- `defaults.yaml` config in `configs/sweep` shows which parameters you want to modify. Note: you could also set all parameters from here (important: if sweep config is used, it will overwrite the same parameters when wandb sweep starts! The pipeline code without sweeps needs some updates and can be used in the future too).
- `test` value of the `wandb.tag` config can be any string value of your choice. 

See other configs in `configs/defaults.yaml`. E.g.you can also set custom paths to choose where to save the results using `custom_path_to_save_data` parameter.




## Acknowledgements

Note: This repo is based on the fork https://github.com/alexaatm/deep-spectral-segmentation, which is our modified version of the original work of https://github.com/lukemelas/deep-spectral-segmentation, credits go to the respective authors who laid the whole foundation of what we were building on! In our fork, we added additional features, such as: new affinities based on image similarity metrics useful for ultrasound (SSD, MI and many more), new priors for the clustering step (position and shape of segments, support of dinov2 models and custom dino models, preprocessing useful for ultrasound (gaussian blurring, histogram equalisation etc.), and configurable pipeline code for tracking runs in wandb. 

In current repo, the abovementioned fork was merged into the main code, as we wanted to clean it from the unused parts of the original code.

 - [Our Modified Version of Deep Spectral Segmentation](https://github.com/alexaatm/deep-spectral-segmentation)
 - [Original Deep Spectral Methods Work](https://github.com/lukemelas/deep-spectral-segmentation)
