from urllib.request import urlopen
import timm
import torch
import numpy as np
import torch.nn as nn
import os


def count_parameters_in_MB(model):
    if isinstance(model, nn.Module):
        return np.sum(np.prod(v.size()) for v in model.parameters()) / 1e6
    else:
        return np.sum(np.prod(v.size()) for v in model) / 1e6

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


resnet_variant_list = ['resnet50.tv_in1k', 'resnet101.tv_in1k', 'resnet152.tv_in1k']
convnext_variant_list = ['convnext_tiny.fb_in1k', 'convnext_small.fb_in1k', 'convnext_base.fb_in1k', 'convnext_large.fb_in1k'] 
vit_variant_list = ['vit_tiny_patch16_224.augreg_in1k', 'vit_small_patch16_224.augreg_in1k', 'vit_base_patch16_224.augreg_in1k', 'vit_large_patch16_224.augreg_in1k']


model_list = resnet_variant_list + convnext_variant_list + vit_variant_list


for model_name in model_list:

    print(f"\nLoading: {model_name}")

    model = timm.create_model(model_name, pretrained=True)
    p1 = count_parameters(model)
    p2 = count_parameters_in_MB(model)

    print(f"{model_name}: p1={p1} MB / p2={p2} MB")
    del model

