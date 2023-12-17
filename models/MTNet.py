from base import BaseModel

from utils.helpers import initialize_weights, set_trainable
from itertools import chain

''' 
-> ResNet BackBone
'''
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:52:52 2020

@author: 308
"""
from torch import nn

import torch

from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F

from models import resnet_with_ccm, resnet


class MT_Module(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(MT_Module, self).__init__()
        # self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet50', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(resnet, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

    def forward(self, x):
        # x_size = x.size()
        x_0 = self.layer0(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        return x_0, x_1, x_2, x_3, x_4


class MT_Module_new(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, qk_channel, vx_channel, blinear=False):
        super(MT_Module_new, self).__init__()
        self.blinear = blinear  # 插值参数

        self.query_conv = Conv2d(in_channels=qk_channel, out_channels=qk_channel // 4, kernel_size=1)
        self.key_conv = Conv2d(in_channels=qk_channel, out_channels=qk_channel // 4, kernel_size=1)
        self.value_conv = Conv2d(in_channels=vx_channel, out_channels=vx_channel, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x, x_h):
        """
            inputs :
                x : 低维特征  h高维特征
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x_h.size()
        b, c, h, w = x.size()

        proj_query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(b, -1, h * w)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        if self.blinear == False:
            proj_value = self.value_conv(x_h).view(m_batchsize, -1, width * height)
            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out = out.view(m_batchsize, C, height, width)
            out = self.gamma * out + x_h
            return out
        else:
            x_h = F.interpolate(x_h, size=(h, w), mode='bilinear', align_corners=True)
            proj_value = self.value_conv(x_h).view(m_batchsize, -1, h * w)
            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out = out.view(m_batchsize, C, h, w)
            out = self.gamma * out + x_h
            return out


class Decoder_new(nn.Module):
    def __init__(self, x_3_channels, x_2_channels, x_1_channels, num_classes):
        super(Decoder_new, self).__init__()
        """升级思路： 1先做交叉自注意力 2拼接 3降维  假设输入的c是512"""
        self.csam_x3 = MT_Module_new(x_3_channels, 512)  # 交叉自注意力 参数1低维特征 参数2 高维特征
        self.x3_reduce = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.x3_down = nn.Sequential(nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))

        self.csam_x2 = MT_Module_new(x_2_channels, 512, blinear=False)  # 交叉自注意力 参数1低维特征 参数2 高维特征
        self.x2_reduce = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.x2_down = nn.Sequential(nn.Conv2d(512 + 256, 384, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(384), nn.ReLU(inplace=True))

        self.csam_x1 = MT_Module_new(x_1_channels, 384, blinear=True)  # 交叉自注意力 参数1低维特征 参数2 高维特征
        self.x1_reduce = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128),
                                       nn.ReLU(inplace=True))
        self.x1_down = nn.Sequential(nn.Conv2d(128 + 384, 256, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.last = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Conv2d(128, num_classes, kernel_size=1, bias=False)
        )

    def forward(self, x, x_3, x_2, x_1):
        csam3 = self.csam_x3(x_3, x)
        x3_reduce = self.x3_reduce(x_3)
        x3 = self.x3_down(torch.cat((csam3, x3_reduce), 1))

        csam2 = self.csam_x2(x_2, x3)
        x2_reduce = self.x2_reduce(x_2)
        x2 = self.x2_down(torch.cat((csam2, x2_reduce), 1))

        csam1 = self.csam_x1(x_1, x2)
        x1_reduce = self.x1_reduce(x_1)
        x1 = self.x1_down(torch.cat((csam1, x1_reduce), 1))

        x0 = self.last(x1)

        return x0, x1, x2, x3  #


class MTNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='resnet', pretrained=True,
                 output_stride=16, freeze_bn=False, freeze_backbone=False, **_):

        super(MTNet, self).__init__()
        assert ('xception' or 'resnet' in backbone)
        if 'resnet' in backbone:
            self.backbone = ResNet(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained, backbone='resnet101')
            x_3_channels = 1024
            x_2_channels = 512
            x_1_channels = 256

        self.conv_1 = nn.Sequential(nn.Conv2d(2048, 512, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True))

        self.attention = MT_Module(in_dim=512)
        self.decoder = Decoder_new(1024, 512, 256, num_classes)
        initialize_weights(self.conv_1, self.attention, self.decoder)
        if freeze_bn: self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.backbone], False)

    def forward(self, inputss):
        H, W = inputss.size(2), inputss.size(3)
        x_0, x_1, x_2, x_3, x_4 = self.backbone(inputss)

        x = self.conv_1(x_4)
        x = self.attention(x)
        x_0_output, x_1_output, x_2_output, x_3_output = self.decoder(x, x_3, x_2, x_1)  # ,
        x0_output = F.interpolate(x_0_output, size=(H, W), mode='bilinear', align_corners=True)
        x1_output = F.interpolate(x_1_output, size=(H, W), mode='bilinear', align_corners=True)
        x2_output = F.interpolate(x_2_output, size=(H, W), mode='bilinear', align_corners=True)
        x3_output = F.interpolate(x_3_output, size=(H, W), mode='bilinear', align_corners=True)
        return tuple([x0_output, x1_output, x2_output, x3_output])  # x1_output, #x3_output

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.conv_1.parameters(), self.decoder.parameters(), self.attention.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


if __name__ == '__main__':
    inputs = torch.randn((2, 3, 224, 224))
    net = MTNet(num_classes=21)
    o1 = net(inputs)
    print(o1[0].size())
    print(o1[1].size())
    print(o1[2].size())