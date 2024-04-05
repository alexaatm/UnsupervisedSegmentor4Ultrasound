import hydra
from omegaconf import DictConfig, OmegaConf
import os
import logging
import wandb

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
def pipeline(cfg: DictConfig) -> None:
    log.info("Starting the pipeline...")

    # Set the directories
    
    # use default local data
    main_data_dir = os.path.join(hydra.utils.get_original_cwd(), '../data')

    if cfg.custom_path_to_save_data!="":
        path_to_save_data = cfg.custom_path_to_save_data
    else:
        custom_path=(f"seg{cfg.segments_num}"
                        f"_clust{cfg.clusters_num}"
                        f"_norm-{cfg.norm}"
                        f"_prepr-{cfg.preprocessed_data}"
                        f"_dino{cfg.spectral_clustering.C_dino}"
                        f"_ssdknn{cfg.spectral_clustering.C_ssd_knn}"
                        f"_var{cfg.spectral_clustering.C_var_knn}"
                        f"_pos{cfg.spectral_clustering.C_pos_knn}"
                        f"_nn{cfg.spectral_clustering.max_knn_neigbors}"
                        f"_ssd{cfg.spectral_clustering.C_ssd}"
                        f"_ncc{cfg.spectral_clustering.C_ncc}"
                        f"_lncc{cfg.spectral_clustering.C_lncc}"
                        f"_ssim{cfg.spectral_clustering.C_ssim}"
                        f"_mi{cfg.spectral_clustering.C_mi}"
                        f"_sam{cfg.spectral_clustering.C_sam}"
                        f"_p{cfg.spectral_clustering.patch_size}"
                        f"_sigma{cfg.spectral_clustering.aff_sigma}"
                        )
        path_to_save_data = os.path.join(os.getcwd(),custom_path)


    # Directories
    if cfg.dataset.dataset_root is not None:
        images_list = os.path.join(main_data_dir, cfg.dataset.dataset_root, cfg.dataset.list)
        dataset_dir = os.path.join(main_data_dir, cfg.dataset.dataset_root)
        if cfg.dataset.gt_dir is not None:
            gt_dir = os.path.join(main_data_dir, cfg.dataset.dataset_root,cfg.dataset.gt_dir)
        if cfg.preprocessed_data=="mixed":
            images_root = os.path.join(main_data_dir, cfg.dataset.dataset_root, cfg.dataset.preprocessed_dir)
        elif cfg.preprocessed_data=="derained":
            images_root = os.path.join(main_data_dir, cfg.dataset.dataset_root, cfg.dataset.derained_dir)
        else:
            images_root = os.path.join(main_data_dir, cfg.dataset.dataset_root, cfg.dataset.images_root)

    else:
        images_list = os.path.join(main_data_dir, cfg.dataset.name, 'lists', 'images.txt')
        dataset_dir = os.path.join(main_data_dir, cfg.dataset.name)
        if cfg.dataset.gt_dir is not None:
            gt_dir = os.path.join(main_data_dir, cfg.dataset.name,cfg.dataset.gt_dir)
        if cfg.preprocessed_data=="mixed":
            images_root = os.path.join(main_data_dir, cfg.dataset.name, cfg.dataset.preprocessed_dir)
        elif cfg.preprocessed_data=="derained":
            images_root = os.path.join(main_data_dir, cfg.dataset.dataset_root, cfg.dataset.derained_dir)
        else:
            images_root = os.path.join(main_data_dir, cfg.dataset.name, cfg.dataset.images_root)



    # Set default output directories
    if cfg.dataset.features_dir is not None:
        output_feat_dir = os.path.join(main_data_dir, cfg.dataset.dataset_root, cfg.dataset.features_dir)
    else:
        output_feat_dir = os.path.join(path_to_save_data, 'features', cfg.model.name)
    output_eig_dir = os.path.join(path_to_save_data, 'eig', cfg.spectral_clustering.which_matrix)
    output_multi_region_seg = os.path.join(path_to_save_data, 'multi_region_segmentation', cfg.spectral_clustering.which_matrix)
    output_bbox = os.path.join(path_to_save_data, 'multi_region_bboxes', cfg.spectral_clustering.which_matrix, 'bboxes.pth')
    output_bbox_features = os.path.join(path_to_save_data, 'multi_region_bboxes', cfg.spectral_clustering.which_matrix, 'bbox_features.pth')
    output_bbox_crops = os.path.join(path_to_save_data, 'multi_region_bboxes', cfg.spectral_clustering.which_matrix, 'bbox_crops')
    output_bbox_clusters = os.path.join(path_to_save_data, 'multi_region_bboxes', cfg.spectral_clustering.which_matrix, 'bbox_clusters.pth')
    output_segmaps = os.path.join(path_to_save_data, 'semantic_segmentations', cfg.spectral_clustering.which_matrix, 'segmaps')
    output_crf_segmaps = os.path.join(path_to_save_data, 'semantic_segmentations', cfg.spectral_clustering.which_matrix, 'crf_segmaps')
    output_crf_multi_region = os.path.join(path_to_save_data, 'semantic_segmentations', cfg.spectral_clustering.which_matrix, 'crf_multi_region')

    
    if cfg.precomputed.mode == "precomputed":
        log.info("Some precomputed steps are provided - need to check each step")
        # check the next step only if the previous step is provided (since snext steps depends on the previous one)
        if cfg.precomputed.features != "" and cfg.dataset.features_dir == "" :
            output_feat_dir = cfg.precomputed.features
        if cfg.precomputed.eig != "":
            output_eig_dir = cfg.precomputed.eig
            if cfg.precomputed.multi_region_segmentation != "":
                output_multi_region_seg = cfg.precomputed.multi_region_segmentation
                if cfg.precomputed.multi_region_segmentation != "":
                    output_crf_multi_region = cfg.precomputed.crf_multi_region                  
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
    log.info(f'output_multi_region_seg={output_multi_region_seg}')
    log.info(f'output_bbox={output_bbox}')
    log.info(f'output_bbox_features={output_bbox_features}')
    log.info(f'output_bbox_crops={output_bbox_crops}')
    log.info(f'output_bbox_clusters={output_bbox_clusters}')
    log.info(f'output_segmaps={output_segmaps}')
    log.info(f'output_crf_segmaps={output_crf_segmaps}')
    log.info(f'output_crf_multi_region={output_crf_multi_region}')


    # Extract Features
    log.info("STEP 1/8: extract features")

    if not cfg.pipeline_steps.dino_features:
        log.info("Step was not selected")
        if cfg.spectral_clustering.C_dino > 0:
            log.info("cfg.spectral_clustering.C_dino=",cfg.spectral_clustering.C_dino)
            log.info("Dino features were selected. Set cfg.pipeline_steps.dino_features to True")
            exit()
    else:
        im_transform_data = extract.extract_features(
            images_list = images_list,
            images_root = images_root,
            output_dir = output_feat_dir,
            model_name = cfg.model.name,
            batch_size = cfg.loader.batch_size,
            model_checkpoint=cfg.model.checkpoint,
            only_dict = True if cfg.spectral_clustering.C_dino == 0.0 else False,
            norm = cfg.norm,
            inv = cfg.inv,
            gauss_blur=cfg.gauss_blur,
            gauss_teta=cfg.gauss_teta,
            hist_eq=cfg.hist_eq
        )

        # Visualize Dino Attention Maps
        if cfg.vis.dino_attn_maps:
            log.info("Plot dino attention maps")
            output_dino_plots = os.path.join(path_to_save_data, 'plots', 'dino_attn_maps')
            vis_utils.plot_dino_attn_maps(
                images_list = images_list,
                images_root = images_root,
                model_checkpoint=cfg.model.checkpoint,
                model_name = cfg.model.name,
                output_dir = output_dino_plots
            )

    # Compute Eigenvectors - spectral clustering step
    log.info("STEP 2/8: extract eigenvectors (spectral clustering)")

    if not cfg.pipeline_steps.eigen:
        log.info("Step was not selected")
        exit()

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
        C_ssd_knn = cfg.spectral_clustering.C_ssd_knn,
        C_dino = cfg.spectral_clustering.C_dino,
        max_knn_neigbors = cfg.spectral_clustering.max_knn_neigbors,
        C_var_knn = cfg.spectral_clustering.C_var_knn,
        C_pos_knn = cfg.spectral_clustering.C_pos_knn, 
        C_ssd = cfg.spectral_clustering.C_ssd,
        C_ncc = cfg.spectral_clustering.C_ncc,
        C_lncc = cfg.spectral_clustering.C_lncc,
        C_ssim = cfg.spectral_clustering.C_ssim,
        C_mi = cfg.spectral_clustering.C_mi,
        C_sam = cfg.spectral_clustering.C_sam,
        patch_size = cfg.spectral_clustering.patch_size,
        aff_sigma = cfg.spectral_clustering.aff_sigma,
        distance_weight1 = cfg.spectral_clustering.distance_weight1,
        distance_weight2 = cfg.spectral_clustering.distance_weight2,
        image_transform_data = im_transform_data,
        use_transform = cfg.spectral_clustering.use_transform,
    )

    # Visualize eigenvectors
    if cfg.vis.eigen:
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

    if not cfg.pipeline_steps.segments:
        log.info("Step was not selected")
        exit()

    extract.extract_multi_region_segmentations(
        features_dir = output_feat_dir,
        eigs_dir = output_eig_dir,
        output_dir = output_multi_region_seg,
        adaptive = cfg.multi_region_segmentation.adaptive,
        non_adaptive_num_segments = cfg.multi_region_segmentation.non_adaptive_num_segments,
        infer_bg_index = cfg.multi_region_segmentation.infer_bg_index,
        kmeans_baseline = cfg.multi_region_segmentation.kmeans_baseline,
        num_eigenvectors = cfg.multi_region_segmentation.num_eigenvectors,
        multiprocessing = cfg.multi_region_segmentation.multiprocessing
    )

    # Visualize multi-region segmentations
    if cfg.vis.multiregion_segmaps:
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
    if not cfg.pipeline_steps.crf_multi_region:
        log.info("Step was not selected")
        # exit()
    else:
        extract.extract_crf_segmentations(
            images_list = images_list,
            images_root = images_root,
            segmentations_dir = output_multi_region_seg,
            output_dir = output_crf_multi_region,
            features_dir = output_feat_dir,
            num_classes =  cfg.multi_region_segmentation.non_adaptive_num_segments, #change to num_segments
            downsample_factor = cfg.crf.downsample_factor,
            multiprocessing = cfg.crf.multiprocessing,
            image_transform_data = im_transform_data,
            # CRF parameters
            w1 = cfg.crf.w1,
            alpha = cfg.crf.alpha,
            beta = cfg.crf.beta,
            w2 = cfg.crf.w2,
            gamma = cfg.crf.gamma,
            it= cfg.crf.it
        )

        # Visualize final crf segmentations
        if cfg.vis.crf_multi_region:
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

    if not cfg.pipeline_steps.bbox:
        log.info("Step was not selected")
        # exit()
    else:

        extract.extract_bboxes(
            features_dir = output_feat_dir,
            segmentations_dir = output_multi_region_seg,
            output_file = output_bbox,
            num_erode = cfg.bbox.num_erode,
            num_dilate = cfg.bbox.num_dilate,
            skip_bg_index= cfg.bbox.skip_bg_index,
            downsample_factor = cfg.bbox.downsample_factor,
        )



        # Extract bounding box features
        log.info("STEP 5/8: extract bounding box features ")

        if not cfg.pipeline_steps.bbox_features:
            log.info("Step was not selected")
            # exit()
        else:
            extract.extract_bbox_features(
                images_root = images_root,
                bbox_file = output_bbox,
                model_name = cfg.model.name,
                output_file = output_bbox_features,
                image_transform_data = im_transform_data,
                output_dir_crops = output_bbox_crops,
                C_pos = cfg.bbox.C_pos,
                C_mask = cfg.bbox.C_mask,
            )


            # Extract clusters
            log.info("STEP 6/8: extract clusters ")

            if not cfg.pipeline_steps.clusters:
                log.info("Step was not selected")
                # exit()
            else:
                extract.extract_bbox_clusters(
                    bbox_features_file = output_bbox_features,
                    output_file = output_bbox_clusters,
                    num_clusters = cfg.bbox.num_clusters,
                    seed = cfg.bbox.seed,
                    pca_dim = cfg.bbox.pca_dim
                )


                # Create semantic segmentations
                log.info("STEP 7/8: create semantic segmentation ")

                if not cfg.pipeline_steps.sem_segm:
                    log.info("Step was not selected")
                    # exit()
                else:
                    extract.extract_semantic_segmentations(
                        segmentations_dir = output_multi_region_seg,
                        bbox_clusters_file = output_bbox_clusters,
                        output_dir = output_segmaps
                    )

                    # Visualize segmentations
                    if cfg.vis.segmaps:
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

    if not cfg.pipeline_steps.crf_segm:
        log.info("Step was not selected")
        exit()

    extract.extract_crf_segmentations(
        images_list = images_list,
        images_root = images_root,
        segmentations_dir = output_segmaps if cfg.pipeline_steps.sem_segm else output_multi_region_seg,
        output_dir = output_crf_segmaps,
        features_dir = output_feat_dir,
        image_transform_data = im_transform_data,
        num_classes =  cfg.crf.num_classes,
        downsample_factor = cfg.crf.downsample_factor,
        multiprocessing = cfg.crf.multiprocessing,
        # CRF parameters
        w1 = cfg.crf.w1,
        alpha = cfg.crf.alpha,
        beta = cfg.crf.beta,
        w2 = cfg.crf.w2,
        gamma = cfg.crf.gamma,
        it= cfg.crf.it
    )

    # Visualize final crf segmentations
    if cfg.vis.crf_segmaps:
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
    if cfg.pipeline_steps.eval:
        log.info("EVALUATION (1/2) (crf_multi_region)")
        evaluate(cfg, dataset_dir=dataset_dir, gt_dir=gt_dir, pred_dir=output_crf_multi_region, tag="crf_multi_region")
        log.info("EVALUATION (2/2) (crf_segmaps)")
        evaluate(cfg, dataset_dir=dataset_dir, gt_dir=gt_dir, pred_dir=output_crf_segmaps,  tag="crf_segmaps")
        log.info("EVALUATION (3/4) (multi_region)")
        evaluate(cfg, dataset_dir=dataset_dir, gt_dir=gt_dir, pred_dir=output_multi_region_seg, tag="multi_region")
        log.info("EVALUATION (4/4) (segmaps)")
        evaluate(cfg, dataset_dir=dataset_dir, gt_dir=gt_dir, pred_dir=output_segmaps, tag="segmaps")


        
# Evaluation
def evaluate(cfg: DictConfig, dataset_dir, gt_dir="", pred_dir="", tag=""):

    if cfg.eval.eval_per_image:
        # Evaluate per image

        # Logging
        if cfg.wandb:
            wandb.login(key=cfg.wandb.key)
            cfg.wandb.key=""
            wandb.init(name ="eval_" + cfg.dataset.name + "_" + segm_eval.datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), project=cfg.wandb.setup.project, config=OmegaConf.to_container(cfg), save_code=True,
                       tags=['pipeline_eval',cfg.wandb.tag, tag, 'eval_per_image', "deep_spectral"])
            cfg = DictConfig(wandb.config.as_dict())  # get the config back from wandb for hyperparameter sweeps

        # Configuration
        print(OmegaConf.to_yaml(cfg))
        print(f'Current working directory: {os.getcwd()}')

        # Evaluate
        # if "CAROTID_MIXED" in dataset_dir:
        #     # make square
        #     resize = transforms.Resize((cfg.dataset.input_size, cfg.dataset.input_size))
        # else:
        #     # only resize
        #     resize = transforms.Resize(cfg.dataset.input_size)
        # dataset = segm_eval.EvalDataset(dataset_dir, gt_dir, pred_dir, transform = resize)
        dataset = segm_eval.EvalDataset(dataset_dir, gt_dir, pred_dir)
        eval_stats, matches, corrected_matches, preds = segm_eval.evaluate_dataset_with_remapping(dataset, cfg.dataset.n_classes, cfg.eval.iou_thresh, cfg.eval.void_label)
        print("eval stats:", eval_stats)
        print("matches:", matches)


        # Visualize some image evaluation samples
        segm_eval.random.seed(1) 
        if cfg.eval.vis_rand_k > len(dataset): # a safeguard for a case when more samples are asked than is in dataset
            inds_to_vis = [0]
        elif cfg.eval.vis_rand_k > 0:
            inds_to_vis = segm_eval.random.sample(range(len(dataset)), cfg.eval.vis_rand_k)
        else:
            inds_to_vis = [-1] #no images will be sampled

        print("sample images to visualize in wandb: ", inds_to_vis)

        # Visualize
        img_list, legend = segm_eval.visualize(dataset, inds_to_vis, cfg.eval.vis_dir)

        if cfg.wandb:
            wandb.log({'mIoU': eval_stats['mIoU']})
            wandb.log({'mIoU_std': eval_stats['mIoU_std']})
            wandb.log({'Pixel_Accuracy': eval_stats['Pixel_Accuracy']})
            wandb.log({'Pixel_Accuracy_std': eval_stats['Pixel_Accuracy_std']})
            wandb.log({'Dice': eval_stats['Dice']})
            wandb.log({'Dice_std': eval_stats['Dice_std']})
            wandb.log({'Precision': eval_stats['Precision']})
            wandb.log({'Precision_std': eval_stats['Precision_std']})
            wandb.log({'Recall': eval_stats['Recall']})
            wandb.log({'Recall_std': eval_stats['Recall_std']})

            # Log metrics and sample evaluation results
            class_names_all = [f'GT_class{i}' for i in range(cfg.dataset.n_classes)]
           
            # Table for logging segment matching
            match_table = wandb.Table(columns = ['ID'] + class_names_all)
    
            # Table for logging corrected labels using wandb
            remapped_pred_table = wandb.Table(columns=['ID', 'Image'])

            # Go through dataset and log data
            for i, (sample, iou_m, match, remapped_pred) in enumerate(zip(dataset, eval_stats['IoU_matrix'], corrected_matches, preds)):
                im, target, pred, metadata = sample
                id = metadata['id']

                # log matches for every sample in dataset - in a table
                row_data = [id] + [None] * cfg.dataset.n_classes
                for pr, gt in match:
                    row_data[1 + gt] = pr if row_data[1 + gt] is None else row_data[1 + gt]
                match_table.add_data(*row_data)

                # log IoU heatmaps and remapped preds only for selected samples
                if i in inds_to_vis:
                    # get lists of unique labels to name the columns of heatmaps (may be different for each image)
                    class_names = [f'GT_class{i}' for i in np.unique(target)]
                    pseudolabel_names = [f'PL_class{i}' for i in np.unique(pred)]

                    # log IoU heatmaps - individually (cannot log wandb heatmaps in a wandb table...)
                    iou_df = segm_eval.pd.DataFrame(data=iou_m, index=pseudolabel_names, columns=class_names)
                    heatmap = wandb.plots.HeatMap(class_names,pseudolabel_names, iou_df, show_text=True)
                    wandb.log({f"Sample IoU Heatmap {id}": heatmap})

                    # log remapped predictions - in a table
                    mask_img = wandb.Image(im, masks = {
                    "groud_truth" : {"mask_data" : target},
                    "prediction" : {"mask_data" : pred},
                    "remapped_pred" : {"mask_data" : remapped_pred},
                    })
                    remapped_pred_table.add_data(id, mask_img)

            # Log completed tables
            wandb.log({"match_table": match_table})
            wandb.log({"Example Images After Remapping" : remapped_pred_table})

            # Log Jaccard index table
            wandb.log({"jaccard_table": wandb.Table(data=[eval_stats['jaccards_all_categs']], columns=class_names_all[1:])})

            # Log example images with overlayed segmentation
            img_table = wandb.Table(columns=['ID', 'Image', 'Pred', 'Ground_Truth'])
            for img_id, img, pred, gt in img_list:
                img_table.add_data(img_id, wandb.Image(img), wandb.Image(pred), wandb.Image(gt))
            wandb.log({"Example Images" : img_table})

            # Log legend to interprete example images
            wandb.log({"Legend Class-Color": wandb.Image(legend)})

            wandb.finish()

    if cfg.eval.eval_per_dataset:
        # Evaluate over the whole dataset

        # Logging
        if cfg.wandb:
            wandb.login(key=cfg.wandb.key)
            cfg.wandb.key=""
            wandb.init(name ="eval_" + cfg.dataset.name + "_" + segm_eval.datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), project=cfg.wandb.setup.project, config=OmegaConf.to_container(cfg), save_code=True,
                tags=['pipeline_eval', cfg.wandb.tag, tag, 'eval_per_dataset', "deep_spectral"])
            cfg = DictConfig(wandb.config.as_dict())  # get the config back from wandb for hyperparameter sweeps

        # Configuration
        print(OmegaConf.to_yaml(cfg))
        print(f'Current working directory: {os.getcwd()}')

        # Evaluate
        if "CAROTID_MIXED" in dataset_dir:
            # make square
            resize = transforms.Resize((cfg.dataset.input_size, cfg.dataset.input_size))
        else:
            # only resize
            resize = transforms.Resize(cfg.dataset.input_size)
        dataset = segm_eval.EvalDataset(dataset_dir, gt_dir, pred_dir, transform = resize)
        eval_stats, match = segm_eval.evaluate_dataset_with_single_matching(dataset, cfg.dataset.n_classes, cfg.dataset.get('n_clusters', None), cfg.eval.iou_thresh)
        # print(eval_stats)
        # print(match)

        # Visualize some image evaluation samples
        segm_eval.random.seed(1)
        if cfg.eval.vis_rand_k > len(dataset): # a safeguard for a case when more samples are asked than is in dataset
            inds_to_vis = [0]
        elif cfg.eval.vis_rand_k > 0:
            inds_to_vis = segm_eval.random.sample(range(len(dataset)), cfg.eval.vis_rand_k)
        else:
            inds_to_vis = [-1] #no images will be sampled

        print("sample images to visualize in wandb: ", inds_to_vis)
        img_list, legend = segm_eval.visualize(dataset, inds_to_vis, cfg.eval.vis_dir)


        if cfg.wandb:
            wandb.log({'mIoU': eval_stats['mIoU']})
            wandb.log({'Pixel_Accuracy': eval_stats['Pixel_Accuracy']})
            wandb.log({'Dice': eval_stats['Dice']})
            wandb.log({'Precision': eval_stats['Precision']})
            wandb.log({'Recall': eval_stats['Recall']})

            # Log confusion matrix and other metrics to wandb
            class_names_all = [f'GT_class{i}' for i in range(cfg.dataset.n_classes)]
            if dataset.n_clusters is not None:
                n_clusters = dataset.n_clusters
            elif cfg.dataset.n_clusters is not None:
                n_clusters = cfg.dataset.n_clusters
            else:
                n_clusters = cfg.dataset.n_classes
            pseudolabel_names = [f'PL_class{i}' for i in range(n_clusters)]
            iou_df = segm_eval.pd.DataFrame(data=eval_stats['IoU_matrix'], index=pseudolabel_names, columns=class_names_all)
            wandb.log({'IoU_heatmap': wandb.plots.HeatMap(class_names_all, pseudolabel_names, iou_df, show_text=True)})

            # Log Jaccard index table
            wandb.log({"jaccard_table": wandb.Table(data=[eval_stats['jaccards_all_categs']], columns=class_names_all)})
            wandb.log({"pix_acc_table": wandb.Table(data=[eval_stats['pix_acc_all_categs']], columns=class_names_all)})
            wandb.log({"dice_table": wandb.Table(data=[eval_stats['dice_all_categs']], columns=class_names_all)})
            wandb.log({"precision_table": wandb.Table(data=[eval_stats['precision_all_categs']], columns=class_names_all)})
            wandb.log({"recall_table": wandb.Table(data=[eval_stats['recall_all_categs']], columns=class_names_all)})


            # Log segment matchings
            pred_gt_list = [[pr, gt] for pr, gt in match]
            wandb.log({"match_table": wandb.Table(data=pred_gt_list, columns=['Pseudo label','Ground Truth label'])})

            # Log example images
            img_table = wandb.Table(columns=['ID', 'Image', 'Pred', 'Ground_Truth'])
            for img_id, img, pred, gt in img_list:
                img_table.add_data(img_id, wandb.Image(img), wandb.Image(pred), wandb.Image(gt))
            wandb.log({"Example Images" : img_table})

            # Log legend to interprete example images
            wandb.log({"Legend CLass-Color": wandb.Image(legend)})

            wandb.finish()
 
# Evaluation only pipeline
def eval_pipeline(cfg: DictConfig) -> None:
    log.info("Starting the evaluation-only pipeline...")

    # Set the directories - use default local data
    main_data_dir = os.path.join(hydra.utils.get_original_cwd(), '../data')

    if cfg.custom_path_to_save_data!="":
        path_to_save_data = cfg.custom_path_to_save_data
    else:
        path_to_save_data = os.path.join(os.getcwd())


    # Directories
    if cfg.dataset.dataset_root is not None:
        dataset_dir = os.path.join(main_data_dir, cfg.dataset.dataset_root)
        if cfg.dataset.gt_dir is not None:
            gt_dir = os.path.join(main_data_dir, cfg.dataset.dataset_root,cfg.dataset.gt_dir)

    else:
        dataset_dir = os.path.join(main_data_dir, cfg.dataset.name)
        if cfg.dataset.gt_dir is not None:
            gt_dir = os.path.join(main_data_dir, cfg.dataset.name,cfg.dataset.gt_dir)

    


    # Set default output directories
    output_crf_multi_region = os.path.join(path_to_save_data, 'semantic_segmentations', cfg.spectral_clustering.which_matrix, 'crf_multi_region')
    output_crf_segmaps = os.path.join(path_to_save_data, 'semantic_segmentations', cfg.spectral_clustering.which_matrix, 'crf_segmaps')
    output_segmaps = os.path.join(path_to_save_data, 'semantic_segmentations', cfg.spectral_clustering.which_matrix, 'segmaps')
    output_multi_region = os.path.join(path_to_save_data, 'multi_region_segmentations', cfg.spectral_clustering.which_matrix)


    # evaluate
    log.info("EVALUATION (1/2) (crf_multi_region)")
    evaluate(cfg, dataset_dir=dataset_dir, gt_dir=gt_dir, pred_dir=output_crf_multi_region, tag="crf_multi_region")
    log.info("EVALUATION (2/2) (crf_segmaps)")
    evaluate(cfg, dataset_dir=dataset_dir, gt_dir=gt_dir, pred_dir=output_crf_segmaps, tag="crf_segmaps")
    log.info("EVALUATION (3/4) (multi_region)")
    evaluate(cfg, dataset_dir=dataset_dir, gt_dir=gt_dir, pred_dir=output_multi_region, tag="multi_region")
    log.info("EVALUATION (4/4) (segmaps)")
    evaluate(cfg, dataset_dir=dataset_dir, gt_dir=gt_dir, pred_dir=output_segmaps, tag="segmaps")


# Pipeline for deep spectral segmentation steps
def vis_pipeline(cfg: DictConfig) -> None:
    log.info("Starting the visualisaion...")

    # Set the directories
    main_data_dir = os.path.join(hydra.utils.get_original_cwd(), '../data')
    path_to_save_data = os.path.join(os.getcwd())


    # Directories
    images_list = os.path.join(main_data_dir, cfg.dataset.name, 'lists', 'images.txt')
    images_root = os.path.join(main_data_dir, cfg.dataset.name, cfg.dataset.images_root)
    log.info(f'images_list={images_list}')
    log.info(f'images_root={images_root}')

    if cfg.precomputed.mode == "precomputed":
        log.info("Some precomputed steps are provided - need to check each step")
        # check the next step only if the previous step is provided (since snext steps depends on the previous one)
        if cfg.precomputed.features != "":
            output_feat_dir = cfg.precomputed.features
            log.info(f'feat_dir={output_feat_dir}')
            if cfg.precomputed.eig != "":
                output_eig_dir = cfg.precomputed.eig
                log.info(f'eig_dir={output_eig_dir}')
                if cfg.precomputed.multi_region_segmentation != "":
                    output_multi_region_seg = cfg.precomputed.multi_region_segmentation
                    log.info(f'multi_region_seg={output_multi_region_seg}')
                    if cfg.precomputed.bboxes != "":
                        output_bbox = cfg.precomputed.bboxes
                        log.info(f'bbox={output_bbox}')
                        if cfg.precomputed.segmaps != "":
                            output_segmaps = cfg.precomputed.segmaps
                            log.info(f'segmaps={output_segmaps}')
                            if cfg.precomputed.crf_segmaps != "":
                                output_crf_segmaps = cfg.precomputed.crf_segmaps
                                log.info(f'crf_segmaps={output_crf_segmaps}')

    # VISUALIZATIONS

    if cfg.vis.eigen:
        log.info("Plot eigenvectors")
        output_eig_plots = os.path.join(path_to_save_data, 'plots', 'eigen')
        vis_utils.plot_eigenvectors(
            images_list = images_list,
            images_root = images_root,
            eigenvevtors_dir = output_eig_dir,
            features_dir = output_feat_dir,
            output_dir = output_eig_plots
        )

    if cfg.vis.dino_attn_maps:
        log.info("Plot dino attention maps")
        output_dino_plots = os.path.join(path_to_save_data, 'plots', 'dino_attn_maps')
        vis_utils.plot_dino_attn_maps(
            images_list = images_list,
            images_root = images_root,
            model_checkpoint=cfg.model.checkpoint,
            model_name = cfg.model.name,
            output_dir = output_dino_plots
        )

    if cfg.vis.crf_segmaps:
        log.info("Plot final segmentations")
        output_segm_plots = os.path.join(path_to_save_data, 'plots', 'crf_segmaps')
        vis_utils.plot_segmentation(
            images_list = images_list,
            images_root = images_root,
            segmentations_dir = output_crf_segmaps,
            bbox_file = output_bbox,
            output_dir = output_segm_plots
        )

    if cfg.vis.multiregion_segmaps:
        log.info("Plot multi region segmentations")
        output_segm_plots = os.path.join(path_to_save_data, 'plots', 'multiregion_segmaps')
        vis_utils.plot_segmentation(
            images_list = images_list,
            images_root = images_root,
            segmentations_dir = output_multi_region_seg,
            bbox_file = None,
            output_dir = output_segm_plots
        )
    
    # Visualize segmentations
    if cfg.vis.segmaps:
        log.info("Plot segmentations")
        output_segm_plots = os.path.join(path_to_save_data, 'plots', 'segmaps')
        vis_utils.plot_segmentation(
            images_list = images_list,
            images_root = images_root,
            segmentations_dir = output_segmaps,
            bbox_file = None,
            output_dir = output_segm_plots
        )
    
@hydra.main(version_base=None, config_path="../../configs", config_name="defaults")
def run_pipeline(cfg: DictConfig) -> None:
    # print(f'cfg.wandb.mode is={cfg.wandb.mode}')

    # if cfg.wandb.mode=='server':
        # login to wandb using locally stored key, remove the key to prevent it from being logged
        # wandb.login(key=cfg.wandb.key)
        # cfg.wandb.key=""
        
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Current working directory  : {}".format(os.getcwd()))

    # wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # run = wandb.init(config=wandb_config, project = cfg.wandb.setup.project, settings=wandb.Settings(start_method='thread'))
    
    if cfg.only_vis:
        vis_pipeline(cfg)
    elif cfg.only_eval:
        eval_pipeline(cfg)
    else:
        pipeline(cfg)    

    # wandb.finish()



if __name__ == "__main__":
    run_pipeline()