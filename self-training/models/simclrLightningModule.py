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
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(512, 2048, 2048)
        # self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)
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
        # TODO: consider decay of the learning rate
        optim = torch.optim.SGD(self.parameters(), lr=0.001) 
        return optim