import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from lightly.models.modules import SimCLRProjectionHead

class SimCLRTriplet(pl.LightningModule):
    def __init__(self, max_epochs=1, lr = 0.001):
        super().__init__()

        self.max_epochs = max_epochs
        self.lr=lr
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)
        # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
        self.criterion = torch.nn.TripletMarginLoss()

    def forward(self, anchor, pos, neg):
        x_anchor = self.backbone(anchor).flatten(start_dim=1)
        z_anchor = self.projection_head(x_anchor)

        x_pos = self.backbone(pos).flatten(start_dim=1)
        z_pos = self.projection_head(x_pos)

        x_neg = self.backbone(neg).flatten(start_dim=1)
        z_neg = self.projection_head(x_neg)

        return z_anchor, z_pos, z_neg

    def _common_step(self, batch, mode='train'):
        (anchor, pos, neg), _, _ = batch
        z_anchor, z_pos, z_neg = self.forward(anchor, pos, neg)
        loss = self.criterion(z_anchor, z_pos, z_neg)
        self.log(f'{mode}_loss', loss)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, mode='val')

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.max_epochs
        )

        return [optim], [scheduler]