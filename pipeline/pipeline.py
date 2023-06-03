import hydra
from omegaconf import DictConfig, OmegaConf
import os
import logging
import wandb

from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path

from extract import extract


# A logger for this file
log = logging.getLogger(__name__)

# Pipeline for deep spectral segmentation steps
def pipeline(cfg: DictConfig) -> None:
    log.info("Starting the pipeline...")

    # Set the directories
    if cfg.wandb.mode=='server':
        # use polyaxon paths
        main_data_dir = os.path.join(get_data_paths()['data1'], '3D_US_vis', 'datasets')
        path_to_save_data = os.path.join(get_outputs_path(), cfg.dataset.name)
    else:
        # use default local data
        main_data_dir = os.path.join(hydra.utils.get_original_cwd(), '../data')
        path_to_save_data = os.path.join(os.getcwd())


    # Directories
    images_list = os.path.join(main_data_dir, cfg.dataset.name, 'lists', 'images.txt')
    images_root = os.path.join(main_data_dir, cfg.dataset.name, 'images')

    # Set default output directories
    output_feat_dir = os.path.join(path_to_save_data, 'features', cfg.model.name)
    output_eig_dir = os.path.join(path_to_save_data, 'eig', cfg.spectral_clustering.which_matrix)
    output_seg_dir = os.path.join(path_to_save_data, 'multi_region_segmentation', cfg.spectral_clustering.which_matrix)
    output_bbox = os.path.join(path_to_save_data, 'multi_region_bboxes', cfg.spectral_clustering.which_matrix, 'bboxes.pth')
    output_bbox_features = os.path.join(path_to_save_data, 'multi_region_bboxes', cfg.spectral_clustering.which_matrix, 'bbox_features.pth')
    output_bbox_clusters = os.path.join(path_to_save_data, 'multi_region_bboxes', cfg.spectral_clustering.which_matrix, 'bbox_clusters.pth')
    output_segmaps = os.path.join(path_to_save_data, 'semantic_segmentations', 'patches', cfg.spectral_clustering.which_matrix, 'segmaps')
    output_crf_segmaps = os.path.join(path_to_save_data, 'semantic_segmentations', 'patches', cfg.spectral_clustering.which_matrix, 'crf_segmaps')
    
    if cfg.precomputed.mode == "precomputed":
        log.info("Some precomputed steps are provided - need to check each step")
        # check the next step only if the previous step is provided (since snext steps depends on the previous one)
        if cfg.precomputed.features != "":
            output_feat_dir = cfg.precomputed.features
            if cfg.precomputed.eig != "":
                output_eig_dir = cfg.precomputed.eig
                if cfg.precomputed.multi_region_segmentation != "":
                    output_seg_dir = cfg.precomputed.multi_region_segmentation
                    if cfg.precomputed.bboxes != "":
                        output_bbox = cfg.precomputed.bboxes
                        if cfg.precomputed.bbox_features != "":
                            output_bbox_features = cfg.precomputed.bbox_features
                            if cfg.precomputed.bbox_clusters != "":
                                output_bbox_clusters = cfg.precomputed.bbox_clusters
                                if cfg.precomputed.segmaps != "":
                                    output_segmaps = cfg.precomputed.segmaps
                                    if cfg.precomputed.crf_segmaps != "":
                                        output_crf_segmaps = cfg.precomputed.crf_segmaps

    log.info(f'images_list={images_list}')
    log.info(f'images_root={images_root}')
    log.info(f'output_feat_dir={output_feat_dir}')
    log.info(f'output_eig_dir={output_eig_dir}')
    log.info(f'output_seg_dir={output_seg_dir}')
    log.info(f'output_bbox={output_bbox}')
    log.info(f'output_bbox_features={output_bbox_features}')
    log.info(f'output_bbox_clusters={output_bbox_clusters}')
    log.info(f'output_segmaps={output_segmaps}')
    log.info(f'output_crf_segmaps={output_crf_segmaps}')


    # Extract Features
    log.info("STEP 1/8: extract features")

    extract.extract_features(
        images_list = images_list,
        images_root = images_root,
        output_dir = output_feat_dir,
        model_name = cfg.model.name,
        batch_size = cfg.loader.batch_size
    )



    # Compute Eigenvectors - spectral clustering step
    log.info("STEP 2/8: extract eigenvectors (spectral clustering)")

    # TODO: figure out how to pass data... cannot read from saved directory??? need to copy data from NAS outputs to data1
    # NOTE: Reading from output_feat_dir seems to work... 
    extract.extract_eigs(
        images_root = images_root,
        features_dir = output_feat_dir,
        output_dir = output_eig_dir,
        which_matrix = cfg.spectral_clustering.which_matrix,
        which_color_matrix = cfg.spectral_clustering.which_color_matrix,
        which_features = cfg.spectral_clustering.which_features,
        normalize = cfg.spectral_clustering.normalize,
        threshold_at_zero = cfg.spectral_clustering.threshold_at_zero,
        lapnorm = cfg.spectral_clustering.lapnorm,
        K = cfg.spectral_clustering.K,
        image_downsample_factor = cfg.spectral_clustering.image_downsample_factor,
        image_color_lambda = cfg.spectral_clustering.image_color_lambda,
        multiprocessing = cfg.spectral_clustering.multiprocessing,
        image_ssd_beta = cfg.spectral_clustering.image_ssd_beta,
    )



    # Extract segments
    log.info("STEP 3/8: extract segments ")

    extract.extract_multi_region_segmentations(
        features_dir = output_feat_dir,
        eigs_dir = output_eig_dir,
        output_dir = output_seg_dir,
        adaptive = cfg.multi_region_segmentation.adaptive,
        non_adaptive_num_segments = cfg.multi_region_segmentation.non_adaptive_num_segments,
        infer_bg_index = cfg.multi_region_segmentation.infer_bg_index,
        kmeans_baseline = cfg.multi_region_segmentation.kmeans_baseline,
        num_eigenvectors = cfg.multi_region_segmentation.num_eigenvectors,
        multiprocessing = cfg.multi_region_segmentation.multiprocessing
    )



    # Extract bounding boxes
    log.info("STEP 4/8: extract bounding boxes ")

    extract.extract_bboxes(
        features_dir = output_feat_dir,
        segmentations_dir = output_seg_dir,
        output_file = output_bbox,
        num_erode = cfg.bbox.num_erode,
        num_dilate = cfg.bbox.num_dilate,
        skip_bg_index= cfg.bbox.skip_bg_index,
        downsample_factor = cfg.bbox.downsample_factor,
    )



    # Extract bounding box features
    log.info("STEP 5/8: extract bounding box features ")

    extract.extract_bbox_features(
        images_root = images_root,
        bbox_file = output_bbox,
        model_name = cfg.model.name,
        output_file = output_bbox_features
    )


    # Extract clusters
    log.info("STEP 6/8: extract clusters ")

    extract.extract_bbox_clusters(
        bbox_features_file = output_bbox_features,
        output_file = output_bbox_clusters,
        num_clusters = cfg.bbox.num_clusters,
        seed = cfg.bbox.seed,
        pca_dim = cfg.bbox.pca_dim
    )


    # # Create semantic segmentations
    # log.info("STEP 7/8: create semantic segmentation ")

    # extract.extract_semantic_segmentations(
    #     segmentations_dir = output_seg_dir,
    #     bbox_clusters_file = output_bbox_clusters,
    #     output_dir = output_segmaps
    # )


    # Create crf segmentations (optional)


    # Evaluate degmentation of evaluation is on

    
@hydra.main(version_base=None, config_path="./configs", config_name="defaults")
def run_pipeline(cfg: DictConfig) -> None:
    print(f'cfg.wandb.mode is={cfg.wandb.mode}')

    if cfg.wandb.mode=='server':
        # login to wandb using locally stored key, remove the key to prevent it from being logged
        wandb.login(key=cfg.wandb.key)
        cfg.wandb.key=""
        
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Current working directory  : {}".format(os.getcwd()))

    # wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # run = wandb.init(config=wandb_config, project = cfg.wandb.setup.project, settings=wandb.Settings(start_method='thread'))
    
    pipeline(cfg)    

    # wandb.finish()



if __name__ == "__main__":
    run_pipeline()