@echo off

REM Example parameters for the semantic segmentation experiments
set "DATASET=liver2_mini"
set "MODEL=dino_vits8"
set "MATRIX=laplacian"
set "DOWNSAMPLE=16"
set "N_SEG=15"
set "N_ERODE=2"
set "N_DILATE=5"
set "DATA_ROOT=../../data"


python ..\extract\extract.py extract_features ^
--images_list "%DATA_ROOT%/%DATASET%/lists/images.txt" ^
--images_root "%DATA_ROOT%/%DATASET%/images" ^
--output_dir "%DATA_ROOT%/%DATASET%/features/%MODEL%" ^
--model_name "%MODEL%" ^
--batch_size 1