Create a config .yaml file for your dataset, following the structure below. E.g. thyroid.yaml:

```
name: thyroid_test9
dataset_root: THYROID
dataset_type: folders
images_root: images
list: lists/images.txt
gt_dir: labels
pred_dir: null
n_classes: 2
features_dir: null
preprocessed_dir: null
derained_dir: null
eigenseg_dir: null
```