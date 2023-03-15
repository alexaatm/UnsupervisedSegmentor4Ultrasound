import copy
import torch
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad
import torchvision
from torch import nn


class DINO(torch.nn.Module):
    def __init__(self, backbone, input_dim):
        super().__init__()
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z
    
def get_dino_backbone(dino_model_name: str):
    backbone = torch.hub.load('facebookresearch/dino:main', dino_model_name, pretrained=False)
    input_dim = backbone.embed_dim
    return (backbone, input_dim)

def get_resnet_backbone():
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    input_dim = 512
    return (backbone, input_dim)

