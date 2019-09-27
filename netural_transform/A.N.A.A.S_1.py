import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
from PIL import Image
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import shutil
from glob import glob

# from tensorboardX import SummaryWriter

import numpy as np
import multiprocessing

import copy
from tqdm import tqdm
from collections import defaultdict

import torch.utils.data.distributed

# from utils import *
from models import *
import time

from pprint import pprint
display = pprint



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

is_hvd = False
tag = 'nohvd'
base = 32
style_weight = 50
content_weight = 1
tv_weight = 1e-6
epochs = 22

batch_size = 8
width = 256

verbose_hist_batch = 100
verbose_image_batch = 800

model_name = f'metanet_base{base}_style{style_weight}_tv{tv_weight}_tag{tag}'
print(f'model_name: {model_name}, rank: {hvd.rank()}')




# class VGG(nn.Module):
#     def __init__(self,transform_features):
#         super(VGG,self).__init__()
#         self.transform_features=transform_features
#         self.layer_name_mapping={
#             '3':"relu1_2",
#             '8':"relu2_2",
#             '15':"relu3_3",
#             '22':"relu4_3"
#         }
#         for p in self.parameters():
#             p.requires_grad=False
#
#     def forward(self, x):
#         outs=[]
#         for name,module in self.transform_features._modules.items():
#             x=module(x)
#             if name in self.layer_name_mapping:
#                 outs.append(x)
#         return outs


# class TransformNet(nn.Module):
#     def __init__(self, base=8):
#         super(TransformNet, self).__init__()
#         self.base = base
#         self.weights = []
#         self.downsampling = nn.Sequential(
#             *ConvLayer(3, base, kernel_size=9, trainable=True),
#             *ConvLayer(base, base * 2, kernel_size=3, stride=2),
#             *ConvLayer(base * 2, base * 4, kernel_size=3, stride=2),
#         )
#         self.residuals = nn.Sequential(*[ResidualBlock(base * 4) for i in range(5)])
#         self.upsampling = nn.Sequential(
#             *ConvLayer(base * 4, base * 2, kernel_size=3, upsample=2),
#             *ConvLayer(base * 2, base, kernel_size=3, upsample=2),
#             *ConvLayer(base, 3, kernel_size=9, instance_norm=False, relu=False, trainable=True),
#         )
#         self.get_param_dict()
#
#     def forward(self, X):
#         y = self.downsampling(X)
#         y = self.residuals(y)
#         y = self.upsampling(y)
#         return y
#
#     def get_param_dict(self):
#         """找出该网络所有 MyConv2D 层，计算它们需要的权值数量"""
#         param_dict = defaultdict(int)
#
#         def dfs(module, name):
#             for name2, layer in module.named_children():
#                 dfs(layer, '%s.%s' % (name, name2) if name != '' else name2)
#             if module.__class__ == MyConv2D:
#                 param_dict[name] += int(np.prod(module.weight.shape))
#                 param_dict[name] += int(np.prod(module.bias.shape))
#
#         dfs(self, '')
#         return param_dict
#
#     def set_my_attr(self, name, value):
#         # 下面这个循环是一步步遍历类似 residuals.0.conv.1 的字符串，找到相应的权值
#         target = self
#         for x in name.split('.'):
#             if x.isnumeric():
#                 target = target.__getitem__(int(x))
#             else:
#                 target = getattr(target, x)
#
#         # 设置对应的权值
#         n_weight = np.prod(target.weight.shape)
#         target.weight = value[:n_weight].view(target.weight.shape)
#         target.bias = value[n_weight:].view(target.bias.shape)
#
#     def set_weights(self, weights, i=0):
#         """输入权值字典，对该网络所有的 MyConv2D 层设置权值"""
#         for name, param in weights.items():
#             self.set_my_attr(name, weights[name][i])


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    transform_features = y.view(b, ch, w * h)
    features_t = transform_features.transpose(1, 2)
    gram = transform_features.bmm(features_t) / (ch * h * w)
    return gram



# 1.load data

#归一化参数
cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]
tensor_normalizer = transforms.Normalize(mean=cnn_normalization_mean, std=cnn_normalization_std)

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(width, scale=(256/480, 1), ratio=(1, 1)),
    transforms.ToTensor(),
    tensor_normalizer
])


style_dataset = torchvision.datasets.ImageFolder('/home/ypw/WikiArt/', transform=data_transform)
content_dataset = torchvision.datasets.ImageFolder('/home/ypw/COCO/', transform=data_transform)

content_data_loader = torch.utils.data.DataLoader(content_dataset, batch_size=batch_size,
    shuffle=True, num_workers=multiprocessing.cpu_count())











vgg16=models.vgg16(pretrained=True)
vgg16=VGG(vgg16.transform_features[:23]).to(device).eval()


n_batch = len(content_data_loader)

with tqdm(enumerate(content_data_loader), total=n_batch) as pbar:
    for batch, (input_img, _) in pbar:
        content_images = content_images.to(device)
        transformed_images =TransformNet(content_images)

        transform_features = vgg16(input_img)
        content_features = vgg16(input_img)
        transformed_features = vgg16(transformed_images)

        run = [0]
        while run[0] <= 300:
            def f():
                optim.zero_grad()
                transform_features = vgg16(input_img)

                content_loss = F.mse_loss(transform_features[2], content_features[2]) * content_weight
                style_loss = 0

                style_gram=[gram_matrix(x) for x in s]
                transform_grams = [gram_matrix(x) for x in transform_features]
                for a, b in zip(transform_grams, style_grams):
                    style_loss += F.mse_loss(a, b) * style_weight

                loss = style_loss + content_loss

                if run[0] % 50 == 0:
                    print('Step {}: Style Loss: {:4f} Content Loss: {:4f}'.format(
                        run[0], style_loss.item(), content_loss.item()))
                run[0] += 1

                loss.backward()
                return loss


            optim.step(f)