from archetectures.vgg import VGG
from archetectures.resnet import *
from archetectures.mobilenetv2 import MobileNetV2
from archetectures.effecientnet import EfficientNetB0

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

map_arg = {
    "resnet18" : ResNet18(),
    "resnet34" : ResNet34(),
    "resnet50" : ResNet50(),
    "resnet101" : ResNet101(),
    "resnet152" : ResNet152(),
    'vgg11' : VGG('VGG11'),
    'vgg16' : VGG('VGG16'), 
    'vgg19' : VGG('VGG19'), 
    'mobilenetv2' : MobileNetV2(),
    'efficientnet' : EfficientNetB0(),
    'adam' : optim.Adam,
    'rmsprop' : optim.RMSprop,
    'sgd' : optim.SGD,
    'cosine' : lr_scheduler.CosineAnnealingLR, 
    'linear' : lr_scheduler.LinearLR, 
    'step' : lr_scheduler.StepLR, 
    'none' : None
}