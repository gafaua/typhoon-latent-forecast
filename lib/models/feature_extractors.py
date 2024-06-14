import ssl

import torch
import torch.nn as nn
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)
from torchvision.models.vgg import VGG11_BN_Weights, vgg11_bn

from lib.models.networks.simple_cnn import SimpleCNN
from lib.models.networks.vision_transformer import vit_base, vit_small, vit_tiny
from lib.models.siamese_ema import SiameseEMA

ssl._create_default_https_context = ssl._create_unverified_context


def _load_checkpoint(model, path):
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["model_dict"])

    print("="*100)
    print(f"Loading model from checkpoint {path}")
    print("="*100)

    return model


def _wrap_model_1to3channels(model):
    return nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=1, bias=False),
        model
    )


def get_resnet18():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    return model


def get_resnet18_3channels():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    return model


def get_moco_encoder(backbone: str, weights: str, dataparallel=False):
    model = SiameseEMA(
            base_encoder=get_feature_extractor(backbone),
            out_dim=384 if backbone == "vit_small" else 512,
    )
    if dataparallel:
        model = nn.DataParallel(model)
    model = _load_checkpoint(model, weights)
    if dataparallel:
        model = model.module.encoder_q[0]
    else:
        model = model.encoder_q[0]
    model.eval()

    return model


def get_encoder(backbone: str, weights: str):
    model = get_feature_extractor(backbone)
    model = _load_checkpoint(model, weights)
    model.eval()

    return model


def get_resnet34():
    model = resnet34(weights=None)
    model.fc = nn.Identity()

    return _wrap_model_1to3channels(model)

def get_resnet50():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()

    return _wrap_model_1to3channels(model)

def get_vgg11():
    model = vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)
    #print(model.classifier[-1])
    del model.classifier[-1]
    return _wrap_model_1to3channels(model)


_feature_extractors = dict(
    vit_tiny=vit_tiny,
    vit_small=vit_small,
    vit_base=vit_base,
    resnet18=get_resnet18,
    resnet18_3c=get_resnet18_3channels,
    resnet34=get_resnet34,
    resnet50=get_resnet50,
    vgg11=get_vgg11,
)


def get_feature_extractor(name):
    return _feature_extractors[name]()
