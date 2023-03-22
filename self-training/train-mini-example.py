from models import dino
import torch
from lightly.data import DINOCollateFunction, LightlyDataset
from lightly.loss import DINOLoss
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import logging

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Current working directory  : {}".format(os.getcwd()))

    # model
    backbone, input_dim = dino.get_dino_backbone("dino_vits16")
    model = dino.DINO(backbone, input_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


    # data
    # dataset = LightlyDataset("../data/liver2_mini/train")
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


if __name__ == "__main__":
    train()