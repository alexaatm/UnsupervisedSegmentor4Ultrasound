import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from lightly.models.modules import SimCLRProjectionHead

class SimCLRTriplet(pl.LightningModule):
    def __init__(self, backbone, hidden_dim, max_epochs=1, lr = 0.001, optimizer="Adam", triplet_loss_margin=1):
        super().__init__()

        self.optimizer_choice=optimizer
        self.max_epochs = max_epochs
        self.lr=lr
        self.backbone=backbone
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 512)
        # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
        self.criterion = torch.nn.TripletMarginLoss(margin=triplet_loss_margin)

    def forward(self, anchor, pos, neg):
        x_anchor = self.backbone(anchor).flatten(start_dim=1)
        z_anchor = self.projection_head(x_anchor)

        x_pos = self.backbone(pos).flatten(start_dim=1)
        z_pos = self.projection_head(x_pos)

        x_neg = self.backbone(neg).flatten(start_dim=1)
        z_neg = self.projection_head(x_neg)

        return z_anchor, z_pos, z_neg

    def _common_step(self, batch, mode='train'):
        (anchor, _, _), (pos, _, _,), (neg, _, _) = batch #unpack the batch, lower dash_ stands for targets and fnames
        z_anchor, z_pos, z_neg = self.forward(anchor, pos, neg)
        loss = self.criterion(z_anchor, z_pos, z_neg)
        self.log(f'{mode}_loss', loss)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, mode='val')

    def configure_optimizers(self):
        if self.optimizer_choice == "sdg":
            optim = torch.optim.SGD(
                self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, self.max_epochs
            )
            return [optim], [scheduler]
        elif self.optimizer_choice == "Adam":
            optim = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, self.max_epochs
            )
            return [optim], [scheduler]
        else:
            raise NotImplementedError()