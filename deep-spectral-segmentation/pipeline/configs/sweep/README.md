Create a sweep config, if you wish to experiment with multiple settings and their combinations (list them using commas). You can use any configs from the pipeline/configs folder and specify the values, using the following structure:

```
name: defaults
seg_for_eval: ['multi_region']
method: grid
count: 1
simple: True
config:
    # generic
    segments_num: [15]
    clusters_num: [6]

    # preprocessing
    norm: ["imagenet"]
    inv: [False]
    hist_eq: [True]
    gauss_blur: [True]

    # affinities
    spectral_clustering:
        C_dino: [1.0]
        C_ssd: [0.0]
        C_mi: [0.0]

    # postprocessing (CRF)
    crf:
        num_classes: [6]
        w1: [15]
        alpha: [7]
        beta: [10]
        w2: [5]
        gamma: [5]
        it: [10]

    # pipeline configs
    
```
