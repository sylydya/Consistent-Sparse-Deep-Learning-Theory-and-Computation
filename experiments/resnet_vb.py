from __future__ import absolute_import

'''Resnet for cifar dataset. 
Ported form 
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei 
'''
import torch.nn as nn
import math
import torch.nn.functional as F

from vb_net import VB_Linear, VB_Conv2d, VB_BatchNorm2d
import numpy as np

__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride=1, lambda_n = 0.01, sigma_0 = 0.00001, sigma_1 = 0.01):
    "3x3 convolution with padding"
    return VB_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, lambda_n = lambda_n, sigma_0=sigma_0, sigma_1=sigma_1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, lambda_n = 0.01, sigma_0 = 0.00001, sigma_1 = 0.01):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, lambda_n = lambda_n, sigma_0=sigma_0, sigma_1=sigma_1)
        self.bn1 = VB_BatchNorm2d(planes, lambda_n = lambda_n, sigma_0=sigma_0, sigma_1=sigma_1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, lambda_n = lambda_n, sigma_0=sigma_0, sigma_1=sigma_1)
        self.bn2 = VB_BatchNorm2d(planes, lambda_n = lambda_n, sigma_0=sigma_0, sigma_1=sigma_1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, lambda_n = 0.01, sigma_0 = 0.00001, sigma_1 = 0.01):
        super(Bottleneck, self).__init__()
        self.conv1 = VB_Conv2d(inplanes, planes, kernel_size=1, bias=False, lambda_n = lambda_n, sigma_0=sigma_0, sigma_1=sigma_1)
        self.bn1 = VB_BatchNorm2d(planes, lambda_n = lambda_n, sigma_0=sigma_0, sigma_1=sigma_1)
        self.conv2 = VB_Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, lambda_n = lambda_n, sigma_0=sigma_0, sigma_1=sigma_1)
        self.bn2 = VB_BatchNorm2d(planes, lambda_n = lambda_n, sigma_0=sigma_0, sigma_1=sigma_1)
        self.conv3 = VB_Conv2d(planes, planes * 4, kernel_size=1, bias=False, lambda_n = lambda_n, sigma_0=sigma_0, sigma_1=sigma_1)
        self.bn3 = VB_BatchNorm2d(planes * 4, lambda_n = lambda_n, sigma_0=sigma_0, sigma_1=sigma_1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class ResNet_sparse_VB(nn.Module):

    def __init__(self, depth, num_classes=1000, lambda_n = 0.01, sigma_0 = 0.00001, sigma_1 = 0.01):
        super(ResNet_sparse_VB, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=44 else BasicBlock

        self.lambda_n = lambda_n
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
        self.c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(sigma_0) - 0.5 * np.log(sigma_1)
        self.c2 = 0.5 / sigma_0 - 0.5 / sigma_1
        self.threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(sigma_1 / sigma_0)) / (
                0.5 / sigma_0 - 0.5 / sigma_1))

        self.inplanes = 16
        self.conv1 = VB_Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False, lambda_n = lambda_n, sigma_0=sigma_0, sigma_1=sigma_1)
        self.bn1 = VB_BatchNorm2d(16, lambda_n = lambda_n, sigma_0=sigma_0, sigma_1=sigma_1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = VB_Linear(64 * block.expansion, num_classes, lambda_n = lambda_n, sigma_0=sigma_0, sigma_1=sigma_1)
        self.prune_flag = 0
        self.mask = None


        for m in self.modules():
            if isinstance(m, VB_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight_mu.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, VB_BatchNorm2d):
                m.weight_mu.data.fill_(1)
                m.bias_mu.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                VB_Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, lambda_n = self.lambda_n, sigma_0=self.sigma_0, sigma_1=self.sigma_1),
                VB_BatchNorm2d(planes * block.expansion, lambda_n = self.lambda_n, sigma_0=self.sigma_0, sigma_1=self.sigma_1),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.prune_flag == 1:
            for name, para in self.named_parameters():
                if 'mu' in name:
                    para.data[self.mask[name]] = 0
                if 'rho' in name:
                    para.data[self.mask[name]] = -float('inf')

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def set_prior(self, lambda_n, sigma_0, sigma_1):
        self.c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(sigma_0) - 0.5 * np.log(sigma_1)
        self.c2 = 0.5 / sigma_0 - 0.5 / sigma_1
        self.threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(sigma_1 / sigma_0)) / (
                0.5 / sigma_0 - 0.5 / sigma_1))
        self.lambda_n = lambda_n
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
        for m in self.modules():
            if hasattr(m, 'set_prior') and m is not self:
                m.set_prior(lambda_n, sigma_0, sigma_1)



    def set_prune(self, user_mask):
        self.mask = user_mask
        self.prune_flag = 1

    def cancel_prune(self):
        self.prune_flag = 0
        self.mask = None



class Lenet5_sparse_VB(nn.Module):
    def __init__(self, lambda_n = 0.01, sigma_0 = 0.00001, sigma_1 = 0.01):
        super(Lenet5_sparse_VB, self).__init__()
        self.conv1 = VB_Conv2d(1, 20, 5, 1, lambda_n = lambda_n, sigma_0=sigma_0, sigma_1=sigma_1)
        self.conv2 = VB_Conv2d(20, 50, 5, 1, lambda_n = lambda_n, sigma_0=sigma_0, sigma_1=sigma_1)
        self.fc1 = VB_Linear(4*4*50, 500, lambda_n = lambda_n, sigma_0=sigma_0, sigma_1=sigma_1)
        self.fc2 = VB_Linear(500, 10, lambda_n = lambda_n, sigma_0=sigma_0, sigma_1=sigma_1)
        self.prune_flag = 0
        self.mask = None

    def forward(self, x):
        if self.prune_flag == 1:
            for name, para in self.named_parameters():
                if 'mu' in name:
                    para.data[self.mask[name]] = 0
                if 'rho' in name:
                    para.data[self.mask[name]] = -float('inf')


        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


    def set_prune(self, user_mask):
        self.mask = user_mask
        self.prune_flag = 1

    def cancel_prune(self):
        self.prune_flag = 0
        self.mask = None



# def resnet(**kwargs):
#     """
#     Constructs a ResNet model.
#     """
#     return ResNet(**kwargs)
