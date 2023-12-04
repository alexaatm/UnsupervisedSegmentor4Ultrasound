Clone here the repository for dinov2: https://github.com/facebookresearch/dinov2.git
in order to modify the needed layers for extracting selfattention based on https://github.com/facebookresearch/dinov2/commit/df7265ce09efa7553a537606565217e42cefea32

Alternatively, clone directly the modified version: https://github.com/3cology/dinov2_with_attention_extraction.git

Note: need to make the directory into modules (put __init__.py file in each folder).
Note2: need to read a checkpoint of the pretrained dinov2 model with modififcations (TODO, see extract_utils.py).


