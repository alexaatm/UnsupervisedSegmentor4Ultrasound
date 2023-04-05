import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

class SimCLR(pl.LightningModule):
    def __init__(self, backbone, hidden_dim, max_epochs=1, lr = 0.001, optimizer="Adam"):
        super().__init__()

        self.optimizer_choice=optimizer
        self.max_epochs = max_epochs
        self.lr=lr
        self.backbone=backbone
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 512)
        self.criterion = NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def _common_step(self, batch, mode='train'):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
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
            return optim
        else:
            raise NotImplementedError()
        
def get_resnet_backbone():
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    hidden_dim = resnet.fc.in_features
    return (backbone, hidden_dim)