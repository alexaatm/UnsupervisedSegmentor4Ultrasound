#!/bin/bash

#! Example parameters for the semantic segmentation experiments
DATASET="liver2_mini"
MODEL="dino_vits8"
MATRIX="laplacian"
DOWNSAMPLE=16
N_SEG=15
N_ERODE=2
N_DILATE=5

python ../extract/extract.py extract_features \
--images_list "./data/${DATASET}/lists/images.txt" \
--images_root "./data/${DATASET}/images" \
--output_dir "./data/${DATASET}/features/${MODEL}" \
--model_name "${MODEL}" \
--batch_size 1