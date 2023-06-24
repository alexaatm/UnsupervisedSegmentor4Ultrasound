from models import dino, dinoLightningModule, simclrLightningModule, simclrTripletLightningModule
from datasets import datasets, samplers
from custom_utils import utils
import torch
from torchvision import transforms
from lightly.data import DINOCollateFunction, LightlyDataset, SimCLRCollateFunction, BaseCollateFunction
from lightly.loss import DINOLoss
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import logging
import wandb

from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path
import shutil

import numpy as np

# A logger for this file
log = logging.getLogger(__name__)

# Dino traing simple example - no saving of checkpoints, wandb moniroting...
def train_dino(cfg: DictConfig) -> None:
    # model
    backbone, input_dim = dino.get_dino_backbone("dino_vits16")
    model = dino.DINO(backbone, input_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


    # data
    dataset = LightlyDataset(cfg.dataset.path)
    collate_fn = DINOCollateFunction()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, #make smaller for dino backbone, was 64 for resnet
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    print(len(dataloader))

    criterion = DINOLoss(
        output_dim=2048,
        warmup_teacher_temp_epochs=5,
    )
    # move loss to correct device because it also contains parameters
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

    log.info("Starting Training")
    for epoch in range(epochs):
        total_loss = 0
        momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
        for views, _, _ in dataloader:
            update_momentum(model.student_backbone, model.teacher_backbone, m=momentum_val)
            update_momentum(model.student_head, model.teacher_head, m=momentum_val)
            views = [view.to(device) for view in views]
            global_views = views[:2]
            teacher_out = [model.forward_teacher(view) for view in global_views]
            student_out = [model.forward(view) for view in views]
            loss = criterion(teacher_out, student_out, epoch=epoch)
            total_loss += loss.detach()
            loss.backward()
            # We only cancel gradients of student head.
            model.student_head.cancel_last_layer_gradients(current_epoch=epoch)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        log.info(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
        wandb.log({"loss": avg_loss}) #assumes wandb was initialized
    
# Dino Lightning Module - convenient for monitoring the training
def train_dinoLightningModule(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.train.seed)

    # data
    if cfg.dataset.name=="imagenet-4-classes":
        transform = transforms.Compose([transforms.Resize(cfg.dataset.input_size),transforms.Grayscale(num_output_channels=3)])
    else:
        transform = transforms.Resize(cfg.dataset.input_size)

    if cfg.wandb.mode=='server':
        # use polyaxon paths
        main_data_dir = os.path.join(get_data_paths()['data1'], '3D_US_vis', 'datasets')
        train_dataset = datasets.PatchDataset(os.path.join(main_data_dir, cfg.dataset.rel_train_path), transform=transform)
        val_dataset = datasets.PatchDataset(os.path.join(main_data_dir, cfg.dataset.rel_val_path), transform=transform)
        path_to_save_ckpt = get_outputs_path()
        
    else:
        # use default local data 
        train_dataset = datasets.PatchDataset(os.path.join(hydra.utils.get_original_cwd(),cfg.dataset.path),transform=transform)
        val_dataset = datasets.PatchDataset(os.path.join(hydra.utils.get_original_cwd(),cfg.dataset.val_path),transform=transform)
        path_to_save_ckpt = os.getcwd()

    log.info(f"Path to save chekpoints: {path_to_save_ckpt}")

    if 'dinov2' in cfg.train.backbone:
        local_crop_size = 98 #to make divisible by 14, dinov2 pacthsize
    else:
        local_crop_size = 96
    
    collate_fn = DINOCollateFunction(
        cj_prob = 0,
        cj_hue = 0, 
        random_gray_scale = 1,
        cj_sat = 0,
        cj_bright=0,
        cj_contrast=0,
        solarization_prob = 0,
        local_crop_size = local_crop_size
    )

    if cfg.loader.mode=="patch":
        # TODO: figure out how to make it a random sampler, cause cannot use shuffle..
        train_sampler=samplers.RandomPatchSampler(
            dataset=train_dataset,
            # patch_mode=cfg.loader.patch_mode,
            patch_size=cfg.loader.patch_size,
            shuffle=True)
        val_sampler=samplers.RandomPatchSampler(
            dataset=val_dataset,
            # patch_mode=cfg.loader.patch_mode,
            patch_size=cfg.loader.patch_size,
            shuffle=False)
    else:
        # need to create a random sampler, because with custom samplers we cannot use shuffle=True in the dataloader
        train_sampler=torch.utils.data.RandomSampler(train_dataset)
        val_sampler=torch.utils.data.SequentialSampler(val_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=cfg.loader.batch_size, #make smaller for dino backbone, was 64 for resnet
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=True,
        num_workers=cfg.loader.num_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=cfg.loader.batch_size, #make smaller for dino backbone, was 64 for resnet
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.loader.num_workers,
    )

    log.info(f"Train dataset: {len(train_dataset)}")
    log.info(f"Val dataset:{len(val_dataset)}")
    log.info(f"Train dataloader: {len(train_dataloader)}")
    log.info(f"Val dataloader: {len(val_dataloader)}")
    log.info(f"Train sampler: {len(train_sampler)}")
    log.info(f"Val sampler: {len(val_sampler)}")


    # model
    if any(x in cfg.train.backbone for x in ('dino_vits16','dino_vits8', 'dinov2_vits14')):
        backbone, input_dim = dinoLightningModule.get_dino_backbone(cfg.train.backbone, cfg.train.pretrained_weights)
    elif cfg.train.backbone=="resnet":
        backbone, input_dim = dinoLightningModule.get_resnet_backbone(cfg.train.pretrained_weights)
    else:
        raise NotImplementedError()
    model = dinoLightningModule.DINO(backbone, input_dim,
        max_epochs=cfg.train.epochs, 
        optimizer=cfg.train.optimizer,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
        )

    # freeze layers if needed
    num_total_blocks = len(model.student_backbone.blocks)
    num_blocks_to_freeze = int (cfg.train.fraction_layers_to_freeze * num_total_blocks)
    
    for i, block in enumerate(model.student_backbone.blocks):
        if i < num_blocks_to_freeze:
            for param in block.parameters():
                param.requires_grad = False
        else:
            break  # Stop freezing layers once the desired number is reached


    # wandb logging
    wandb_logger = pl.loggers.WandbLogger()
    wandb_logger.watch(model)

    #
    class LogDinoInputViewsCallback(pl.Callback):
        def on_train_batch_end(
                self, trainer, pl_module, outputs, batch, batch_idx):
            """Called when the train/val batch ends."""

            # `outputs` comes from `LightningModule.validation_step` or training_step
            # which corresponds to our model predictions in this case

            # Let's log augmented views from the first batch
            if batch_idx == 0:
                views, _, image_names = batch
                global_views = views[:2]

                # take only a single image views and global views
                # each view has 8 v of length batch_size
                single_sample_views = [v[0] for v in views]
                single_sample_global_views = [v[0] for v in views[:2]]

                # log images with `WandbLogger.log_image`
                wandb_logger.log_image(
                    key='views',
                    images=single_sample_views)

                wandb_logger.log_image(
                    key='global_views',
                    images=single_sample_global_views)

    class LogDinoAttentionMapsForCheckpointsCallback(pl.Callback):
        def on_save_checkpoint(
                self, trainer, pl_module, checkpoint):
            """Called when saving checkpoints."""

            # use checkpoint's state-dict for the model
            state_dict = checkpoint['state_dict']
            pl_module.load_state_dict(state_dict, strict=True)
            model = pl_module.teacher_backbone
            model.fc = torch.nn.Identity()
            patch_size = model.patch_embed.patch_size

            # put model to cuda device to
            model = model.to('cuda')

            # Take random 5 samples from the dataset
            rand_val_samples=np.random.choice(len(val_dataset), 5)

            attn_maps = []
            for idx in rand_val_samples:
                # sample =  val_dataset[idx]
                image, _, fname = val_dataset[idx]
                attn_map_plot = utils.extract_attention_map_dino_per_image(model, patch_size, image)
                attn_maps.append(attn_map_plot)

            # log images with `WandbLogger.log_image`
            wandb_logger.log_image(
                key='attention_maps', 
                images=attn_maps)


    # trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            # ModelCheckpoint(save_weights_only=False, mode='min', monitor='val_loss',
            #                 save_top_k=5, filename=path_to_save_ckpt + '/{epoch}-{step}-{val_loss:.2f}'),
            # ModelCheckpoint(every_n_epochs=1, filename=path_to_save_ckpt + '/{epoch}-{step}-{train_loss:.2f}-{val_loss:.2f}'),
            ModelCheckpoint(save_weights_only=True, save_top_k=-1, filename=path_to_save_ckpt + '/{epoch}-{step}-{train_loss:.2f}-{val_loss:.2f}'),
            LearningRateMonitor('epoch'),
            LogDinoInputViewsCallback(),
            EarlyStopping(monitor="val_loss", mode="min", patience = 10, verbose=True),
            LogDinoAttentionMapsForCheckpointsCallback()
        ],
        logger=wandb_logger,
        log_every_n_steps=1,
        )
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # saving the final model
    trainer.save_checkpoint(path_to_save_ckpt + '/final_model.ckpt', weights_only=False)

def train_simclr(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.train.seed)


    # data
    resize = transforms.Resize(cfg.dataset.input_size)
    if cfg.wandb.mode=='server':
        # use polyaxon paths
        main_data_dir = os.path.join(get_data_paths()['data1'], '3D_US_vis', 'datasets')
        train_dataset = LightlyDataset(os.path.join(main_data_dir, cfg.dataset.rel_train_path), transform=resize)
        val_dataset = LightlyDataset(os.path.join(main_data_dir, cfg.dataset.rel_val_path), transform=resize)
        path_to_save_ckpt = get_outputs_path()
    else:
        # use default local data 
        train_dataset = LightlyDataset(os.path.join(hydra.utils.get_original_cwd(),cfg.dataset.path),transform=resize)
        val_dataset = LightlyDataset(os.path.join(hydra.utils.get_original_cwd(),cfg.dataset.val_path),transform=resize)
        path_to_save_ckpt = os.getcwd()


    collate_fn = SimCLRCollateFunction(
        input_size=cfg.dataset.input_size,
        cj_prob = 0.0,
        # gaussian_blur=0.0,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.loader.batch_size,
        collate_fn = collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.loader.num_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.loader.batch_size,
        collate_fn = collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.loader.num_workers,
    )
    

    log.info(f"Train dataset: {len(train_dataset)}")
    log.info(f"Val dataset:{len(val_dataset)}")
    log.info(f"Train dataloader: {len(train_dataloader)}")
    log.info(f"Val dataloader: {len(val_dataloader)}")

    # model
    if cfg.train.backbone=="resnet":
        backbone, hidden_dim = simclrLightningModule.get_resnet_backbone(cfg.train.pretrained_weights)
    else:
        raise NotImplementedError()
    model = simclrLightningModule.SimCLR(
        backbone,
        hidden_dim, 
        max_epochs=cfg.train.epochs, 
        lr=cfg.train.lr,
        optimizer=cfg.train.optimizer)

    # wandb logging
    wandb_logger = pl.loggers.WandbLogger()
    wandb_logger.watch(model)

    #
    class LogSimclrInputViewsCallback(pl.Callback):
        def on_train_batch_end(
                self, trainer, pl_module, outputs, batch, batch_idx):
            """Called when the train/val batch ends."""

            # `outputs` comes from `LightningModule.validation_step` or training_step
            # which corresponds to our model predictions in this case

            # Let's log augmented views from the first batch
            # Take 0th item from each, since when batch is bigger than 1, each x0 and x1
            # will be the length of the batch
            if batch_idx == 0:
                (x0, x1), _, _ = batch

                # log images with `WandbLogger.log_image`
                wandb_logger.log_image(
                    key='x0, x1',
                    images=[x0[0], x1[0]])


    # trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            ModelCheckpoint(save_weights_only=False, mode='min', monitor='val_loss',
                            save_top_k=4, filename=path_to_save_ckpt + '/{epoch}-{step}-{val_loss:.2f}'),
            ModelCheckpoint(every_n_epochs=100, filename=path_to_save_ckpt + '/{epoch}-{step}-{train_loss:.2f}-{val_loss:.2f}'),
            LearningRateMonitor('epoch'),
            LogSimclrInputViewsCallback(),
            EarlyStopping(monitor="val_loss", mode="min", patience = 25, verbose=True)
        ],
        logger=wandb_logger,
        log_every_n_steps=1,
        )
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # saving the final model
    trainer.save_checkpoint(path_to_save_ckpt + '/final_model.ckpt', weights_only=False)

def train_triplet(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.train.seed)

    # data
    # resize = transforms.Resize(cfg.dataset.input_size)
    if cfg.wandb.mode=='server':
        # use polyaxon paths
        main_data_dir = os.path.join(get_data_paths()['data1'], '3D_US_vis', 'datasets')
        if cfg.loader.mode=="patch":
            train_dataset = datasets.TripletPatchDataset(os.path.join(main_data_dir, cfg.dataset.rel_train_path))
            val_dataset = datasets.TripletPatchDataset(os.path.join(main_data_dir, cfg.dataset.rel_val_path))
        else:
            train_dataset = datasets.TripletDataset(os.path.join(main_data_dir, cfg.dataset.rel_train_path), mode = cfg.dataset.triplet_mode)
            val_dataset = datasets.TripletDataset(os.path.join(main_data_dir, cfg.dataset.rel_val_path), mode = cfg.dataset.triplet_mode)
        path_to_save_ckpt = get_outputs_path()
    else:
         # use default local data 
        if cfg.loader.mode=="patch":
            train_dataset = datasets.TripletPatchDataset(os.path.join(hydra.utils.get_original_cwd(),cfg.dataset.path))
            val_dataset = datasets.TripletPatchDataset(os.path.join(hydra.utils.get_original_cwd(),cfg.dataset.val_path))
        else:
            train_dataset = datasets.TripletDataset(os.path.join(hydra.utils.get_original_cwd(),cfg.dataset.path), mode = cfg.dataset.triplet_mode)
            val_dataset = datasets.TripletDataset(os.path.join(hydra.utils.get_original_cwd(),cfg.dataset.val_path), modoe = cfg.dataset.triplet_mode)
        path_to_save_ckpt = os.getcwd()
    
    print(f'Train dataset sample={train_dataset[0]}')
    # data processing
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if 'carotid' in cfg.dataset.name or "imagenet" in cfg.dataset.name:
        # resize to acquare images (val set has varied sizes...)
        resize = transforms.Resize((cfg.dataset.input_size,cfg.dataset.input_size))
    else:
        resize = transforms.Resize(cfg.dataset.input_size)
    transform = transforms.Compose([transforms.ToTensor(), resize, normalize])

    # transforms for the positive patch in the triplet
    pos_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomResizedCrop(size=cfg.loader.patch_size, scale=(0.4, 1.0)),
                transforms.RandomApply(transforms=[transforms.GaussianBlur(sigma=(0.1, 2), kernel_size=(5, 9))], p=0.5)
            ]
        )
    collate_fn = datasets.TripletBaseCollateFunction(transform, pos_transform=pos_transform)

    if cfg.loader.mode=="patch":
        train_sampler=samplers.TripletPatchSampler(
            dataset=train_dataset,
            patch_size=cfg.loader.patch_size,
            shuffle=True,
            max_shift=cfg.loader.max_shift)
        val_sampler=samplers.TripletPatchSampler(
            dataset=val_dataset,
            patch_size=cfg.loader.patch_size,
            shuffle=False,
            max_shift=cfg.loader.max_shift)
    else:
        train_sampler=torch.utils.data.RandomSampler(train_dataset)
        val_sampler=torch.utils.data.SequentialSampler(val_dataset)


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=cfg.loader.batch_size,
        collate_fn = collate_fn,
        shuffle=False,
        drop_last=True,
        num_workers=cfg.loader.num_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=cfg.loader.batch_size,
        collate_fn = collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.loader.num_workers,
    )

    log.info(f"Train dataset: {len(train_dataset)}")
    log.info(f"Val dataset:{len(val_dataset)}")
    log.info(f"Train dataloader: {len(train_dataloader)}")
    log.info(f"Val dataloader: {len(val_dataloader)}")
    log.info(f"Train sampler: {len(train_sampler)}")
    log.info(f"Val sampler: {len(val_sampler)}")

    # model
    if cfg.train.backbone=="resnet":
        backbone, hidden_dim = simclrLightningModule.get_resnet_backbone(cfg.train.pretrained_weights)
    elif any(x in cfg.train.backbone for x in ('dino_vits16','dino_vits8', 'dinov2_vits14')):
        backbone, hidden_dim = dinoLightningModule.get_dino_backbone(cfg.train.backbone, cfg.train.pretrained_weights)
    else:
        raise NotImplementedError()
    model = simclrTripletLightningModule.SimCLRTriplet(
        backbone,
        hidden_dim, 
        max_epochs=cfg.train.epochs, 
        lr=cfg.train.lr,
        optimizer=cfg.train.optimizer,
        triplet_loss_margin=cfg.train.triplet_loss_margin)

    # wandb logging
    wandb_logger = pl.loggers.WandbLogger()
    wandb_logger.watch(model)

    #
    class LogSimclrInputViewsCallback(pl.Callback):
        def on_train_batch_end(
                self, trainer, pl_module, outputs, batch, batch_idx):
            """Called when the train/val batch ends."""

            # Let's log triplet -  from the first batch
            if batch_idx == 0:
                (anchor, pos, neg) = batch
                # each is of length of the batch, so to get just the image, 
                # need to pass both the index o fthe batch, and the 0th item -
                # will be the image (1st is target, 2nd is fname)

                # log images with `WandbLogger.log_image`
                wandb_logger.log_image(
                    key='anchor, pos, neg',
                    images=[anchor[0][0], pos[0][0], neg[0][0]])


    # trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            ModelCheckpoint(save_weights_only=False, mode='min', monitor='val_loss',
                            save_top_k=4, filename=path_to_save_ckpt + '/{epoch}-{step}-{val_loss:.2f}'),
            ModelCheckpoint(every_n_epochs=100, filename=path_to_save_ckpt + '/{epoch}-{step}-{train_loss:.2f}-{val_loss:.2f}'),
            LearningRateMonitor('epoch'),
            LogSimclrInputViewsCallback(),
            EarlyStopping(monitor="val_loss", mode="min", patience = 25, verbose=True)
        ],
        logger=wandb_logger,
        log_every_n_steps=1,
        )
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # saving the final model
    trainer.save_checkpoint(path_to_save_ckpt + '/final_model.ckpt', weights_only=False)

@hydra.main(version_base=None, config_path="./configs", config_name="defaults")
def run_experiment(cfg: DictConfig) -> None:
    print(f'cfg.wandb.mode is={cfg.wandb.mode}')

    # if cfg.wandb.mode=='server':
    # login to wandb using locally stored key, remove the key to prevent it from being logged
    wandb.login(key=cfg.wandb.key)
    cfg.wandb.key=""
        
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Current working directory  : {}".format(os.getcwd()))

    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    if cfg.experiment.name == "train_dino":
        log.info(f"Experiment chosen: {cfg.experiment.name}")
        run = wandb.init(config=wandb_config, project = cfg.wandb.setup.project, settings=wandb.Settings(start_method='thread'))
        train_dino(cfg)
    elif cfg.experiment.name == "train_dinoLightningModule":
        log.info(f"Experiment chosen: {cfg.experiment.name}")
        run = wandb.init(config=wandb_config, project = cfg.wandb.setup.project, settings=wandb.Settings(start_method='thread'))
        train_dinoLightningModule(cfg)
    elif cfg.experiment.name == "train_simclr":
        log.info(f"Experiment chosen: {cfg.experiment.name}")
        run = wandb.init(config=wandb_config, project = cfg.wandb.setup.project, settings=wandb.Settings(start_method='thread'))
        train_simclr(cfg)
    elif cfg.experiment.name == "train_triplet":
        log.info(f"Experiment chosen: {cfg.experiment.name}")
        run = wandb.init(config=wandb_config, project = cfg.wandb.setup.project, settings=wandb.Settings(start_method='thread'))
        train_triplet(cfg)
    else:
        raise ValueError(f'No experiment called: {cfg.experiment.name}')
    

    if cfg.wandb.mode=='server':
        # take care to copy outputs to polyaxon storage (NAS), because files on node where the code is will be deleted
        utils.copytree(os.getcwd(), get_outputs_path())


    wandb.finish()



if __name__ == "__main__":
    run_experiment()