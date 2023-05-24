import copy

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule


# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.


class DINO(pl.LightningModule):
    def __init__(self, backbone, input_dim, max_epochs=1, optimizer="Adam", lr = 0.001, weight_decay=0):
        super().__init__()

        self.max_epochs=max_epochs
        self.optimizer_choice=optimizer
        self.lr=lr
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)
        self.weight_decay=weight_decay

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, mode='train')
    
    def validation_step(self, batch, batch_idx):
        self._common_step(batch, mode='val')
    
    def _common_step(self, batch, mode='train'):
        momentum = cosine_schedule(self.current_epoch, self.max_epochs, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views, a, b = batch
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        
        self.log(f'{mode}_loss', loss)
        return loss


    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        if self.optimizer_choice=="Adam":
            optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, self.max_epochs
            )
            return [optim], [scheduler]
        else:
            raise NotImplementedError()
    
def get_dino_backbone(dino_model_name: str, pretrained_weights = False):
    if "dinov2" in dino_model_name:
        # eg for dinov2 models like dinov2_vits14
        if pretrained_weights:
            backbone = torch.hub.load('facebookresearch/dinov2:main', dino_model_name, pretrained=True) 
        else:
            backbone = torch.hub.load('facebookresearch/dinov2:main', dino_model_name, pretrained=False)
    else:
        if pretrained_weights:
            backbone = torch.hub.load('facebookresearch/dino:main', dino_model_name, pretrained=True) 
        else:
            backbone = torch.hub.load('facebookresearch/dino:main', dino_model_name, pretrained=False)
    input_dim = backbone.embed_dim
    return (backbone, input_dim)

def get_resnet_backbone(pretrained_weights = False):
    # TODO: change to resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    # unify with the function above
    if pretrained_weights:
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    input_dim = 512
    return (backbone, input_dim)