# polyaxon-specififc libraries
from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path

from models import dino, dinoLightningModule, simclrLightningModule, simclrTripletLightningModule
from datasets import datasets
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

import shutil

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
    main_data_dir = os.path.join(get_data_paths()['data1'], '3D_US_vis', 'datasets')
    resize = transforms.Resize(cfg.dataset.input_size)
    train_dataset = LightlyDataset(os.path.join(main_data_dir, cfg.dataset.rel_train_path), transform=resize)
    val_dataset = LightlyDataset(os.path.join(main_data_dir, cfg.dataset.rel_val_path), transform=resize)

    collate_fn = DINOCollateFunction(
        # input_size=cfg.dataset.input_size, #doesnt work for Dino Collate function, TODO: overwrite DInoFunction with inout size attribute added
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.loader.batch_size, #make smaller for dino backbone, was 64 for resnet
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.loader.num_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
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

    # model
    if any(x in cfg.train.backbone for x in ('dino_vits16','dino_vits8')):
        backbone, input_dim = dinoLightningModule.get_dino_backbone(cfg.train.backbone)
    else:
        raise NotImplementedError()
    model = dinoLightningModule.DINO(backbone, input_dim,
        max_epochs=cfg.train.epochs, 
        optimizer=cfg.train.optimizer,
        lr=cfg.train.lr
        )


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



    # trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            ModelCheckpoint(save_weights_only=False, mode='min', monitor='val_loss',
                            save_top_k=3, filename='{epoch}-{step}-{val_loss:.2f}'),
            ModelCheckpoint(every_n_epochs=100, filename='{epoch}-{step}-{train_loss:.2f}-{val_loss:.2f}'),
            LearningRateMonitor('epoch'),
            LogDinoInputViewsCallback(),
            EarlyStopping(monitor="val_loss", mode="min", patience = 10)
        ],
        logger=wandb_logger,
        log_every_n_steps=1,
        )
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # saving the final model
    trainer.save_checkpoint('final_model.ckpt', weights_only=False)

def train_simclr(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.train.seed)


    # data
    main_data_dir = os.path.join(get_data_paths()['data1'], '3D_US_vis', 'datasets')
    train_dataset = LightlyDataset(os.path.join(main_data_dir, cfg.dataset.rel_train_path))
    val_dataset = LightlyDataset(os.path.join(main_data_dir, cfg.dataset.rel_val_path))

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
        backbone, hidden_dim = simclrLightningModule.get_resnet_backbone()
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
                            save_top_k=3, filename='{epoch}-{step}-{val_loss:.2f}'),
            ModelCheckpoint(every_n_epochs=100, filename='{epoch}-{step}-{train_loss:.2f}-{val_loss:.2f}'),
            LearningRateMonitor('epoch'),
            LogSimclrInputViewsCallback(),
            EarlyStopping(monitor="val_loss", mode="min", patience = 10)
        ],
        logger=wandb_logger,
        log_every_n_steps=1,
        )
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # saving the final model
    trainer.save_checkpoint('final_model.ckpt', weights_only=False)

def train_simclr_triplet(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.train.seed)

    # data
    main_data_dir = os.path.join(get_data_paths()['data1'], '3D_US_vis', 'datasets')
    train_dataset = datasets.TripletDataset(os.path.join(main_data_dir,cfg.dataset.rel_train_path))
    val_dataset = datasets.TripletDataset(os.path.join(main_data_dir,cfg.dataset.rel_val_path))
    
    # data processing
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    resize = transforms.Resize(cfg.dataset.input_size)
    transform = transforms.Compose([transforms.ToTensor(), resize, normalize])
    collate_fn = datasets.TripletBaseCollateFunction(transform)

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
        backbone, hidden_dim = simclrLightningModule.get_resnet_backbone()
    else:
        raise NotImplementedError()
    model = simclrTripletLightningModule.SimCLRTriplet(
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
                            save_top_k=3, filename='{epoch}-{step}-{val_loss:.2f}'),
            ModelCheckpoint(every_n_epochs=100, filename='{epoch}-{step}-{train_loss:.2f}-{val_loss:.2f}'),
            LearningRateMonitor('epoch'),
            LogSimclrInputViewsCallback(),
            EarlyStopping(monitor="val_loss", mode="min", patience = 10)
        ],
        logger=wandb_logger,
        log_every_n_steps=1,
        )
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # saving the final model
    trainer.save_checkpoint('final_model.ckpt', weights_only=False)

@hydra.main(version_base=None, config_path="./configs", config_name="defaults")
def run_experiment(cfg: DictConfig) -> None:
    # login to wandb using locally stored key, remove the key to prevent it from being logged
    wandb.login(key=cfg.wandb.key)
    cfg.wandb.key=""

    log.info(OmegaConf.to_yaml(cfg))
    log.info("Current working directory  : {}".format(os.getcwd()))

    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.login(key=cfg.wandb.key)

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
    elif cfg.experiment.name == "train_simclr_triplet":
        log.info(f"Experiment chosen: {cfg.experiment.name}")
        run = wandb.init(config=wandb_config, project = cfg.wandb.setup.project, settings=wandb.Settings(start_method='thread'))
        train_simclr_triplet(cfg)
    else:
        raise ValueError(f'No experiment called: {cfg.experiment.name}')
    

    # take care to copy outputs to polyaxon storage (NAS), because files on node where the code is will be deleted
    utils.copytree(os.getcwd(), get_outputs_path())


    wandb.finish()



if __name__ == "__main__":
    # Polyaxon
    experiment = Experiment()

    data_paths = get_data_paths()
    outputs_path = get_outputs_path()

    print(f'IN path: {data_paths}')
    print(f'OUT path: {outputs_path}')

    print(f"Current working directory : {os.getcwd()}")

    run_experiment()