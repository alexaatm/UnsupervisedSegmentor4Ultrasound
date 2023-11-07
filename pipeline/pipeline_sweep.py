import hydra
from omegaconf import DictConfig, OmegaConf
import os
import logging
import wandb

from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path

from extract import extract
from extract import extract_utils as utils
from vis import vis_utils

from torchvision import transforms

# for evaluation, need to add thesis-codebase script to the python path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
thesis_codebase_folder = os.path.join(current_dir, '..', '..')
print(thesis_codebase_folder)
sys.path.append(thesis_codebase_folder)

# Now you can use absolute imports in your script
# from evaluation.segm_eval import evaluate_dataset,evaluate_dataset_with_single_matching,visualize
from evaluation import segm_eval
import numpy as np

# A logger for this file
log = logging.getLogger(__name__)

# Pipeline for deep spectral segmentation steps
def pipeline(cfg):
    # cfg - wandb config
    log.info("Starting the pipeline...")

    # Set the directories
    if cfg['wandb']['mode']=='server':
        # use polyaxon paths
        main_data_dir = os.path.join(get_data_paths()['data1'], '3D_US_vis', 'datasets')
        path_to_save_data = os.path.join(get_outputs_path(), cfg['dataset']['name'])
    else:
        # use default local data
        main_data_dir = os.path.join(hydra.utils.get_original_cwd(), '../data')

        if cfg['custom_path_to_save_data']!="":
            path_to_save_data = cfg['custom_path_to_save_data']
        else:
            print(f"DEBUG: wandb tag: {cfg['wandb']['tag']}")
            print(f"DEBUG: is Aff in tag? {'aff' in cfg['wandb']['tag']}")
            print(f"DEBUG: is crf in tag? {'crf' in cfg['wandb']['tag']}")
            if ('Aff' in cfg['wandb']['tag'] or 'aff' in cfg['wandb']['tag']):
                print("DEBUG: 'aff' or 'Aff' is in tag!!")
                custom_path=(f"{cfg['wandb']['tag']}/seg{cfg['segments_num']}"
                            f"_clust{cfg['clusters_num']}"
                            f"_norm-{cfg['norm']}"
                            f"_prepr-{cfg['preprocessed_data']}"
                            f"_dino{cfg['spectral_clustering']['C_dino']}"
                            f"_ssdknn{cfg['spectral_clustering']['C_ssd_knn']}"
                            f"_var{cfg['spectral_clustering']['C_var_knn']}"
                            f"_pos{cfg['spectral_clustering']['C_pos_knn']}"
                            f"_nn{cfg['spectral_clustering']['max_knn_neigbors']}"
                            f"_ssd{cfg['spectral_clustering']['C_ssd']}"
                            f"_ncc{cfg['spectral_clustering']['C_ncc']}"
                            f"_lncc{cfg['spectral_clustering']['C_lncc']}"
                            f"_ssim{cfg['spectral_clustering']['C_ssim']}"
                            f"_mi{cfg['spectral_clustering']['C_mi']}"
                            f"_sam{cfg['spectral_clustering']['C_sam']}"
                            f"_p{cfg['spectral_clustering']['patch_size']}"
                            f"_sigma{cfg['spectral_clustering']['aff_sigma']}"
                            )
            elif ('crf' in cfg['wandb']['tag']):
                print("DEBUG: 'crf' is in tag!!")
                custom_path=(f"{cfg['wandb']['tag']}/seg{cfg['segments_num']}"
                            f"_clust{cfg['clusters_num']}"
                            f"_norm-{cfg['norm']}"
                            f"_prepr-{cfg['preprocessed_data']}"
                            f"_dino{cfg['spectral_clustering']['C_dino']}"
                            f"_ssdknn{cfg['spectral_clustering']['C_ssd_knn']}"
                            f"_CRF_alpha{cfg['crf']['alpha']}"
                            f"_beta{cfg['crf']['beta']}"
                            f"_gamma{cfg['crf']['gamma']}"
                            f"_it{cfg['crf']['it']}"
                            f"_w1{cfg['crf']['w1']}"
                            f"_w2{cfg['crf']['w2']}"                         
                            )
            else:
                custom_path=(f"{cfg['wandb']['tag']}/seg{cfg['segments_num']}"
                            f"_clust{cfg['clusters_num']}"
                            f"_norm-{cfg['norm']}"
                            f"_prepr-{cfg['preprocessed_data']}"
                            f"_dino{cfg['spectral_clustering']['C_dino']}"
                            # f"_time{datetime.now()}"
                            )
            path_to_save_data = os.path.join(os.getcwd(), custom_path)

    # Directories
    if cfg['dataset']['dataset_root'] is not None:
        images_list = os.path.join(main_data_dir, cfg['dataset']['dataset_root'], cfg['dataset']['list'])
        dataset_dir = os.path.join(main_data_dir, cfg['dataset']['dataset_root'])
        if cfg['dataset']['gt_dir'] is not None:
            gt_dir = os.path.join(main_data_dir, cfg['dataset']['dataset_root'],cfg['dataset']['gt_dir'])
        if cfg['preprocessed_data']:
            images_root = os.path.join(main_data_dir, cfg['dataset']['dataset_root'], cfg['dataset']['preprocessed_dir'])
        else:
            images_root = os.path.join(main_data_dir, cfg['dataset']['dataset_root'], cfg['dataset']['images_root'])

    else:
        images_list = os.path.join(main_data_dir, cfg['dataset']['name'], 'lists', 'images.txt')
        dataset_dir = os.path.join(main_data_dir, cfg['dataset']['name'])
        if cfg['dataset']['gt_dir'] is not None:
            gt_dir = os.path.join(main_data_dir, cfg['dataset']['name'],cfg['dataset']['gt_dir'])
        if cfg['preprocessed_data']:
            images_root = os.path.join(main_data_dir, cfg['dataset']['name'], cfg['dataset']['preprocessed_dir'])
        else:
            images_root = os.path.join(main_data_dir, cfg['dataset']['name'], cfg['dataset']['images_root'])



    # Set default output directories
    if cfg['dataset']['features_dir'] is not None:
        output_feat_dir = os.path.join(main_data_dir, cfg['dataset']['dataset_root'], cfg['dataset']['features_dir'])
    else:
        output_feat_dir = os.path.join(path_to_save_data, 'features', cfg['model']['name'])
    output_eig_dir = os.path.join(path_to_save_data, 'eig', cfg['spectral_clustering']['which_matrix'])
    output_multi_region_seg = os.path.join(path_to_save_data, 'multi_region_segmentation', cfg['spectral_clustering']['which_matrix'])
    output_bbox = os.path.join(path_to_save_data, 'multi_region_bboxes', cfg['spectral_clustering']['which_matrix'], 'bboxes.pth')
    output_bbox_features = os.path.join(path_to_save_data, 'multi_region_bboxes', cfg['spectral_clustering']['which_matrix'], 'bbox_features.pth')
    output_bbox_clusters = os.path.join(path_to_save_data, 'multi_region_bboxes', cfg['spectral_clustering']['which_matrix'], 'bbox_clusters.pth')
    output_segmaps = os.path.join(path_to_save_data, 'semantic_segmentations', cfg['spectral_clustering']['which_matrix'], 'segmaps')
    output_crf_segmaps = os.path.join(path_to_save_data, 'semantic_segmentations', cfg['spectral_clustering']['which_matrix'], 'crf_segmaps')
    output_crf_multi_region = os.path.join(path_to_save_data, 'semantic_segmentations', cfg['spectral_clustering']['which_matrix'], 'crf_multi_region')

    
    if cfg['precomputed']['mode'] == "precomputed":
        log.info("Some precomputed steps are provided - need to check each step")
        # check the next step only if the previous step is provided (since snext steps depends on the previous one)
        if cfg['precomputed']['features'] != "" and cfg['dataset']['features_dir'] == "" :
            output_feat_dir = cfg['precomputed']['features']
        if cfg['precomputed']['eig'] != "":
            output_eig_dir = cfg['precomputed']['eig']
            if cfg['precomputed']['multi_region_segmentation'] != "":
                output_multi_region_seg = cfg['precomputed']['multi_region_segmentation']
                if cfg['precomputed']['multi_region_segmentation'] != "":
                    output_crf_multi_region = cfg['precomputed']['crf_multi_region']                  
                    if cfg['precomputed']['bboxes'] != "":
                        output_bbox = cfg['precomputed']['bboxes']
                        if cfg['precomputed']['bbox_features'] != "":
                            output_bbox_features = cfg['precomputed']['bbox_features']
                            if cfg['precomputed']['bbox_clusters'] != "":
                                output_bbox_clusters = cfg['precomputed']['bbox_clusters']
                                if cfg['precomputed']['segmaps'] != "":
                                    output_segmaps = cfg['precomputed']['segmaps']
                                    if cfg['precomputed']['crf_segmaps'] != "":
                                        output_crf_segmaps = cfg['precomputed']['crf_segmaps']

    log.info(f'images_list={images_list}')
    log.info(f'images_root={images_root}')
    log.info(f'output_feat_dir={output_feat_dir}')
    log.info(f'output_eig_dir={output_eig_dir}')
    log.info(f'output_multi_region_seg={output_multi_region_seg}')
    log.info(f'output_bbox={output_bbox}')
    log.info(f'output_bbox_features={output_bbox_features}')
    log.info(f'output_bbox_clusters={output_bbox_clusters}')
    log.info(f'output_segmaps={output_segmaps}')
    log.info(f'output_crf_segmaps={output_crf_segmaps}')
    log.info(f'output_crf_multi_region={output_crf_multi_region}')

    # Set the sweep parameters that have to be the same
    log.info("STEP 0: SWEEP configiration")
    log.info(f"cfg['segments_num']={cfg['segments_num']}")
    log.info(f"cfg['clusters_num']={cfg['clusters_num']}")
    cfg['spectral_clustering']['K'] = cfg['segments_num']
    cfg['multi_region_segmentation']['non_adaptive_num_segments'] = cfg['segments_num']
    cfg['bbox']['num_clusters'] = cfg['clusters_num']
    cfg['crf']['num_classes'] = cfg['clusters_num']

    assert(cfg['multi_region_segmentation']['non_adaptive_num_segments'] == cfg['spectral_clustering']['K'])
    assert(cfg['crf']['num_classes'] == cfg['bbox']['num_clusters'])
    

    # Extract Features
    log.info("STEP 1/8: extract features")

    if not cfg['pipeline_steps']['dino_features']:
        log.info("Step was not selected")
        if cfg['spectral_clustering']['C_dino'] > 0:
            log.info("cfg['spectral_clustering']['C_dino']=",cfg['spectral_clustering']['C_dino'])
            log.info("Dino features were selected. Set cfg['pipeline_steps']['dino_features'] to True")
            exit()
    else:
        extract.extract_features(
            images_list = images_list,
            images_root = images_root,
            output_dir = output_feat_dir,
            model_name = cfg['model']['name'],
            batch_size = cfg['loader']['batch_size'],
            model_checkpoint=cfg['model']['checkpoint'],
            only_dict = True if cfg['spectral_clustering']['C_dino'] == 0.0 else False,
            norm = cfg['norm']
        )

        # Visualize Dino Attention Maps
        if cfg['vis']['dino_attn_maps']:
            log.info("Plot dino attention maps")
            output_dino_plots = os.path.join(path_to_save_data, 'plots', 'dino_attn_maps')
            vis_utils.plot_dino_attn_maps(
                images_list = images_list,
                images_root = images_root,
                model_checkpoint=cfg['model']['checkpoint'],
                model_name = cfg['model']['name'],
                output_dir = output_dino_plots
            )

    # Compute Eigenvectors - spectral clustering step
    log.info("STEP 2/8: extract eigenvectors (spectral clustering)")

    if not cfg['pipeline_steps']['eigen']:
        log.info("Step was not selected")
        exit()

    # TODO: figure out how to pass data... cannot read from saved directory??? need to copy data from NAS outputs to data1
    # NOTE: Reading from output_feat_dir seems to work... 
    extract.extract_eigs(
        images_root = images_root,
        features_dir = output_feat_dir,
        output_dir = output_eig_dir,
        which_matrix = cfg['spectral_clustering']['which_matrix'],
        which_color_matrix = cfg['spectral_clustering']['which_color_matrix'],
        which_features = cfg['spectral_clustering']['which_features'],
        normalize = cfg['spectral_clustering']['normalize'],
        threshold_at_zero = cfg['spectral_clustering']['threshold_at_zero'],
        lapnorm = cfg['spectral_clustering']['lapnorm'],
        K = cfg['spectral_clustering']['K'],
        image_downsample_factor = cfg['spectral_clustering']['image_downsample_factor'],
        image_color_lambda = cfg['spectral_clustering']['image_color_lambda'],
        multiprocessing = cfg['spectral_clustering']['multiprocessing'],
        C_ssd_knn = cfg['spectral_clustering']['C_ssd_knn'],
        C_dino = cfg['spectral_clustering']['C_dino'],
        max_knn_neigbors = cfg['spectral_clustering']['max_knn_neigbors'],
        C_var_knn = cfg['spectral_clustering']['C_var_knn'],
        C_pos_knn = cfg['spectral_clustering']['C_pos_knn'],
        C_ssd = cfg['spectral_clustering']['C_ssd'],
        C_ncc = cfg['spectral_clustering']['C_ncc'],
        C_lncc = cfg['spectral_clustering']['C_lncc'],
        C_ssim = cfg['spectral_clustering']['C_ssim'],
        C_mi = cfg['spectral_clustering']['C_mi'],
        C_sam = cfg['spectral_clustering']['C_sam'],
        patch_size = cfg['spectral_clustering']['patch_size'],
        aff_sigma = cfg['spectral_clustering']['aff_sigma'],
        distance_weight1 = cfg['spectral_clustering']['distance_weight1'],
        distance_weight2 = cfg['spectral_clustering']['distance_weight2']

    )

    # Visualize eigenvectors
    if cfg['vis']['eigen']:
        log.info("Plot eigenvectors")
        output_eig_plots = os.path.join(path_to_save_data, 'plots', 'eigen')
        vis_utils.plot_eigenvectors(
            images_list = images_list,
            images_root = images_root,
            eigenvevtors_dir = output_eig_dir,
            features_dir = output_feat_dir,
            output_dir = output_eig_plots
        )

    # Extract segments
    log.info("STEP 3/8: extract segments ")

    if not cfg['pipeline_steps']['segments']:
        log.info("Step was not selected")
        exit()

    extract.extract_multi_region_segmentations(
        features_dir = output_feat_dir,
        eigs_dir = output_eig_dir,
        output_dir = output_multi_region_seg,
        adaptive = cfg['multi_region_segmentation']['adaptive'],
        non_adaptive_num_segments = cfg['multi_region_segmentation']['non_adaptive_num_segments'],
        infer_bg_index = cfg['multi_region_segmentation']['infer_bg_index'],
        kmeans_baseline = cfg['multi_region_segmentation']['kmeans_baseline'],
        num_eigenvectors = cfg['multi_region_segmentation']['num_eigenvectors'],
        multiprocessing = cfg['multi_region_segmentation']['multiprocessing']
    )

    # Visualize multi-region segmentations
    if cfg['vis']['multiregion_segmaps']:
        log.info("Plot multi region segmentations")
        output_segm_plots = os.path.join(path_to_save_data, 'plots', 'multiregion_segmaps')
        vis_utils.plot_segmentation(
            images_list = images_list,
            images_root = images_root,
            segmentations_dir = output_multi_region_seg,
            bbox_file = None,
            output_dir = output_segm_plots
        )

        # Create crf segmentations (optional)
    log.info("[optional]: create CRF semantic segmentation ")

    # Improve multi-region segmentations using crf
    if not cfg['pipeline_steps']['crf_multi_region']:
        log.info("Step was not selected")
        # exit()
    else:
        extract.extract_crf_segmentations(
            images_list = images_list,
            images_root = images_root,
            segmentations_dir = output_multi_region_seg,
            output_dir = output_crf_multi_region,
            num_classes =  cfg['multi_region_segmentation']['non_adaptive_num_segments'], #change to num_segments
            downsample_factor = cfg['crf']['downsample_factor'],
            multiprocessing = cfg['crf']['multiprocessing'],
            # CRF parameters
            w1 = cfg['crf']['w1'],
            alpha = cfg['crf']['alpha'],
            beta = cfg['crf']['beta'],
            w2 = cfg['crf']['w2'],
            gamma = cfg['crf']['gamma'],
            it= cfg['crf']['it']
        )

        # Visualize final crf segmentations
        if cfg['vis']['crf_multi_region']:
            log.info("Plot crf multi region segmentations")
            output_segm_plots = os.path.join(path_to_save_data, 'plots', 'crf_multi_region')
            vis_utils.plot_segmentation(
                images_list = images_list,
                images_root = images_root,
                segmentations_dir = output_crf_multi_region,
                bbox_file = None,
                output_dir = output_segm_plots
            )



    # Extract bounding boxes
    log.info("STEP 4/8: extract bounding boxes ")

    if not cfg['pipeline_steps']['bbox']:
        log.info("Step was not selected")
        # exit()
    else:

        extract.extract_bboxes(
            features_dir = output_feat_dir,
            segmentations_dir = output_multi_region_seg,
            output_file = output_bbox,
            num_erode = cfg['bbox']['num_erode'],
            num_dilate = cfg['bbox']['num_dilate'],
            skip_bg_index= cfg['bbox']['skip_bg_index'],
            downsample_factor = cfg['bbox']['downsample_factor'],
        )



        # Extract bounding box features
        log.info("STEP 5/8: extract bounding box features ")

        if not cfg['pipeline_steps']['bbox_features']:
            log.info("Step was not selected")
            # exit()
        else:
            extract.extract_bbox_features(
                images_root = images_root,
                bbox_file = output_bbox,
                model_name = cfg['model']['name'],
                output_file = output_bbox_features
            )


            # Extract clusters
            log.info("STEP 6/8: extract clusters ")

            if not cfg['pipeline_steps']['clusters']:
                log.info("Step was not selected")
                # exit()
            else:
                extract.extract_bbox_clusters(
                    bbox_features_file = output_bbox_features,
                    output_file = output_bbox_clusters,
                    num_clusters = cfg['bbox']['num_clusters'],
                    seed = cfg['bbox']['seed'],
                    pca_dim = cfg['bbox']['pca_dim']
                )


                # Create semantic segmentations
                log.info("STEP 7/8: create semantic segmentation ")

                if not cfg['pipeline_steps']['sem_segm']:
                    log.info("Step was not selected")
                    # exit()
                else:
                    extract.extract_semantic_segmentations(
                        segmentations_dir = output_multi_region_seg,
                        bbox_clusters_file = output_bbox_clusters,
                        output_dir = output_segmaps
                    )

                    # Visualize segmentations
                    if cfg['vis']['segmaps']:
                        log.info("Plot semantic segmentations")
                        output_segm_plots = os.path.join(path_to_save_data, 'plots', 'segmaps')
                        vis_utils.plot_segmentation(
                            images_list = images_list,
                            images_root = images_root,
                            segmentations_dir = output_segmaps,
                            bbox_file = None,
                            output_dir = output_segm_plots
                        )

    # Create crf segmentations (optional)
    log.info("STEP 8/8 [optional]: create CRF semantic segmentation ")

    if not cfg['pipeline_steps']['crf_segm']:
        log.info("Step was not selected")
        exit()

    extract.extract_crf_segmentations(
        images_list = images_list,
        images_root = images_root,
        segmentations_dir = output_segmaps if cfg['pipeline_steps']['sem_segm'] else output_multi_region_seg,
        output_dir = output_crf_segmaps,
        num_classes =  cfg['crf']['num_classes'],
        downsample_factor = cfg['crf']['downsample_factor'],
        multiprocessing = cfg['crf']['multiprocessing'],
        # CRF parameters
        w1 = cfg['crf']['w1'],
        alpha = cfg['crf']['alpha'],
        beta = cfg['crf']['beta'],
        w2 = cfg['crf']['w2'],
        gamma = cfg['crf']['gamma'],
        it= cfg['crf']['it']
    )

    # Visualize final crf segmentations
    if cfg['vis']['crf_segmaps']:
        log.info("Plot crf semantic segmentations")
        output_segm_plots = os.path.join(path_to_save_data, 'plots', 'crf_segmaps')
        vis_utils.plot_segmentation(
            images_list = images_list,
            images_root = images_root,
            segmentations_dir = output_crf_segmaps,
            bbox_file = None,
            output_dir = output_segm_plots
        )

    # Evaluate segmentation if evaluation is on
    if cfg['pipeline_steps']['eval']:
        log.info("EVALUATION (1/2) (crf_multi_region) - SKIPPED")
        log.info("EVALUATION (2/2) (crf_segmaps)")
        eval_results = evaluate(cfg, dataset_dir=dataset_dir, gt_dir=gt_dir, pred_dir=output_crf_segmaps,  tag="crf_segmaps")

        return eval_results

# Evaluation
def evaluate(cfg, dataset_dir, gt_dir="", pred_dir="", tag=""):

    if cfg['eval']['eval_per_image']:

        # Configuration
        print(f'Current working directory: {os.getcwd()}')

        # Evaluate
        # resize = transforms.Resize(cfg['dataset']['input_size'])
        dataset = segm_eval.EvalDataset(dataset_dir, gt_dir, pred_dir, transform = None)
        eval_stats, matches, corrected_matches, preds = segm_eval.evaluate_dataset_with_remapping(dataset, cfg['dataset']['n_classes'], cfg['eval']['iou_thresh'], cfg['eval']['void_label'])
        print("eval stats:", eval_stats)
        print("matches:", matches)
    
        # log to wandb
        # wandb.log({'mIoU': eval_stats['mIoU']})
        wandb.log({'mIoU_std': eval_stats['mIoU_std']})
        wandb.log({'Pixel_Accuracy': eval_stats['Pixel_Accuracy']})
        wandb.log({'Pixel_Accuracy_std': eval_stats['Pixel_Accuracy_std']})
        wandb.log({'Dice': eval_stats['Dice']})
        wandb.log({'Dice_std': eval_stats['Dice_std']})
        wandb.log({'Precision': eval_stats['Precision']})
        wandb.log({'Precision_std': eval_stats['Precision_std']})
        wandb.log({'Recall': eval_stats['Recall']})
        wandb.log({'Recall_std': eval_stats['Recall_std']})

         # Table for logging corrected labels using wandb
        remapped_pred_table = wandb.Table(columns=['ID', 'Image'])

        # Visualize some image evaluation samples
        segm_eval.random.seed(1) 
        if cfg['eval']['vis_rand_k'] > len(dataset): # a safeguard for a case when more samples are asked than is in dataset
            inds_to_vis = [0]
        elif cfg['eval']['vis_rand_k'] > 0:
            inds_to_vis = segm_eval.random.sample(range(len(dataset)), cfg['eval']['vis_rand_k'])
        else:
            inds_to_vis = [-1] #no images will be sampled

        # Go through dataset and log data
        for i, (sample, iou_m, match, remapped_pred) in enumerate(zip(dataset, eval_stats['IoU_matrix'], corrected_matches, preds)):
            im, target, pred, metadata = sample
            id = metadata['id']

            if i in inds_to_vis:
                # log remapped predictions - in a table
                mask_img = wandb.Image(im, masks = {
                "groud_truth" : {"mask_data" : target},
                "prediction" : {"mask_data" : pred},
                "remapped_pred" : {"mask_data" : remapped_pred},
                })
                remapped_pred_table.add_data(id, mask_img)

        wandb.log({"Example Images After Remapping" : remapped_pred_table})

        # Log Jaccard index table
        class_names_all = [f'GT_class{i}' for i in range(cfg['dataset']['n_classes'])]
        wandb.log({"jaccard_table": wandb.Table(data=[eval_stats['jaccards_all_categs']], columns=class_names_all[1:])})

        
        return eval_stats

    else:
        return    

def objective(cfg):
    eval_stats = pipeline(cfg)
    mIoU =  eval_stats['mIoU']
    return mIoU

def main():
    print(f"MAIN1 wandb.config={wandb.config}")
    wandb.init(name ="sweepCRF_" + segm_eval.datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
               project="pipeline_eval",
               save_code=True,
               tags=['sweep', 'eval_per_image', "deep_spectral", 'sweep_crf'])
            #    tags=['sweep', 'eval_per_image', "deep_spectral", 'sweep_num_cl'])
    cfg=wandb.config
    print(f"MAIN2 wandb.config={cfg}")
    score = objective(cfg)
    # log the main score 
    wandb.log({"mIoU": score})
   
import yaml

def yaml_to_nested_dict(yaml_data):
    def recurse(data):
        if isinstance(data, dict):
            nested_dict = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    nested_dict[key] = {'parameters': recurse(value)}
                else:
                    nested_dict[key] = {'value': value}
            return nested_dict
        elif isinstance(data, list):
            return [{'parameters': recurse(item)} if isinstance(item, dict) else {'value': item} for item in data]
        else:
            return data
    return recurse(yaml_data)

@hydra.main(version_base=None, config_path="./configs", config_name="defaults")
def run_sweep(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Current working directory  : {}".format(os.getcwd()))

    # log into W&B
    wandb.login(key=cfg.wandb.key)
    cfg.wandb.key=""

    # Define the search space
    sweep_configuration = {
        "method": "bayes",
        "metric": {
            "goal": "maximize",
            "name": "mIoU"
        },
        
    }

    # Add original parameters from hydra config
    # Interpolate values from Hydra config
    cfg = OmegaConf.to_container(cfg, resolve=True)
    yaml_config = OmegaConf.to_yaml(cfg)
    yaml_data = yaml.safe_load(yaml_config)
    parameters_dict=yaml_to_nested_dict(yaml_data)

    # add parameters for the sweep
    # sweep_dict = {
        # carotids
        # "segments_num": {"values": [7, 10, 15,16,17,18,19,20,21,22]},
        # 'clusters_num': {"values": [6, 7, 8, 9, 10, 11, 12,13,14,15]}
        # thyroid
        # "segments_num": {"max":30, "min": 10},
        # "clusters_num": {"values": [7]}
        # "clusters_num": {"values": [6, 7, 8, 9, 10, 11, 12]},
        # liver
        # "segments_num": {"max":30, "min": 8},
        # "clusters_num": {"values": [6]}
        # "clusters_num": {"values": [5, 6, 7, 8, 9, 10]},
    # }

    # sweep_dict = {
        # SWEEP for combining affinity matrices
        # "spectral_clustering":{
        #     'parameters': {
        #         'C_dino': {'values': [0.0, 1.0]},
        #         'C_ssd_knn': {'values': [0.0, 1.0]},
        #         'max_knn_neigbors': {'max': 80, 'min': 8},
        #         'C_var_knn': {'values': [0.0, 1.0]},
        #         'C_pos_knn': {'values': [0.0, 1.0]},
        #         'C_ssd': {'values': [0.0, 1.0]},
        #         'C_ncc': {'values': [0.0, 1.0]},
        #         'C_lncc': {'values': [0.0, 1.0]},
        #         'C_ssim': {'values': [0.0, 1.0]},
        #         'C_mi': {'values': [0.0, 1.0]},
        #         'C_sam': {'values': [0.0, 1.0]},
        #         'patch_size': {'values': [16, 8]},
        #         'aff_sigma': {'max': 10.0, 'min': 0.001},
        #     }
        # }

        # SWEEP for finding good parameters for knn-based affinities (W_ssd_knn)
        # "spectral_clustering":{
        #     'parameters': {
        #         'C_dino': {'values': [0.0]},
        #         'C_ssd_knn': {'values': [1.0]},
        #         'max_knn_neigbors': {'max': 120, 'min': 4},
        #         'C_var_knn': {'values': [0.0]},
        #         'C_pos_knn': {'values': [0.0]},
        #         'C_ssd': {'values': [0.0]},
        #         'C_ncc': {'values': [0.0]},
        #         'C_lncc': {'values': [0.0]},
        #         'C_ssim': {'values': [0.0]},
        #         'C_mi': {'values': [0.0]},
        #         'C_sam': {'values': [0.0]},
        #         'patch_size': {'values': [32, 16, 8]},
        #         # 'aff_sigma': {'max': 10.0, 'min': 0.001},
        #         'distance_weight1': {'max': 10.0, 'min': 1.0},
        #         'distance_weight2': {'max': 5.0, 'min': 0.01},
        #     }
        # }
    # }

    # SWEEP for finding good crf parameters
    sweep_dict = {
        "crf":{
            'parameters': {
                'w1': {'max': 20.0, 'min': 4.0},
                'alpha': {'max': 100.0, 'min': 30.0},
                'beta': {'max': 24.0, 'min': 2.0},
                'w2': {'max': 16.0, 'min': 1.0},
                'gamma': {'max': 6.0, 'min': 1.0},
                'it': {'max': 15.0, 'min': 5.0},
            }
        }
    }

    print(sweep_dict)


    # ref: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb#scrollTo=guKch4YoOcfz
    for key, value in sweep_dict.items():
        if key in parameters_dict:
            if 'parameters' in sweep_dict[key].keys():
                parameters_dict[key]['parameters'].update(value['parameters'])
            else:
                parameters_dict[key] = value

    print(parameters_dict)

    sweep_configuration['parameters'] = parameters_dict
    print(sweep_configuration)

    # Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=cfg['wandb']['setup']['project'])
    wandb.agent(sweep_id, function=main, count=40)
if __name__ == "__main__":
    run_sweep()
    