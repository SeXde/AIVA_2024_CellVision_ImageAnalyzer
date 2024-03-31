import torchvision
import torch
from functools import partial
from torchvision.models.detection.fcos import FCOSClassificationHead


def create_fcos_model(num_classes=91, min_size=640, max_size=640):
    model = torchvision.models.detection.fcos_resnet50_fpn(
        weights='DEFAULT'
    )
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = FCOSClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size
    for param in model.parameters():
        param.requires_grad = True
    return model
