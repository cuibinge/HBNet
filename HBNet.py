#!/usr/bin/python3
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from boundary_org import BoundaryEnhancementModule
import cv2
import os

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            m.initialize()

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out      = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out      = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out      = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(out+residual, inplace=True)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64  #当前层输入通道数为64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) #C:3->64 HW:1/2
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1) #不变
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1) #C:64->128 HW:1/4
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1) #C:128->256 HW:1/8
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1) #C:256->512 HW:1/16
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = None
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))#实际下采样执行代码
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)] #Bottleneck的初始化，这个代码dilation=1，结合前后可删减
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))#追加Bottleneck块，执行剩余n-1次
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True) #C:3->64 HW:1/2
        print("out1 after conv1+bn+relu:", out1.shape)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1) #C:64 HW:1/4
        print("out1 after maxpool:", out1.shape)
        out2 = self.layer1(out1) #C:64->256 HW:1/4
        out3 = self.layer2(out2) #C:256->512 HW:1/8
        out4 = self.layer3(out3) #C:512->1024 HW:1/16
        out5 = self.layer4(out4) #C:1024->2048 HW:1/32
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('resnet50-19c8e357.pth'), strict=False)

# 边缘增强模块：
class BoundaryEnhancementModule(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(BoundaryEnhancementModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        
        # 三种可学习的边界检测算子
        self.learnable_sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                          stride=1, padding=1, groups=in_channels, bias=False)
        self.learnable_sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                          stride=1, padding=1, groups=in_channels, bias=False)
        self.learnable_laplace = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                          stride=1, padding=1, groups=in_channels, bias=False)
        
        # 初始化算子权重（保持原有结构）
        self._init_operators()
        
        # 权重学习网络（替代原来的融合方式）
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 3, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 3, 1),  # 输出3个权重
            nn.Softmax(dim=1)
        )
        
        # 输出转换层（保持通道数一致）
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def _init_operators(self):
        
        # Sobel X
        sobel_x = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.learnable_sobel_x.weight.data = sobel_x.repeat(self.in_channels, 1, 1, 1)
        self.learnable_sobel_x.weight.requires_grad = True
        
        # Sobel Y
        sobel_y = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.learnable_sobel_y.weight.data = sobel_y.repeat(self.in_channels, 1, 1, 1)
        self.learnable_sobel_y.weight.requires_grad = True
        
        # Laplace
        laplace = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        self.learnable_laplace.weight.data = laplace.repeat(self.in_channels, 1, 1, 1)
        self.learnable_laplace.weight.requires_grad = True

    def forward(self, x):
        # 三种边界检测
        sobel_x = torch.abs(self.learnable_sobel_x(x))
        sobel_y = torch.abs(self.learnable_sobel_y(x))
        laplace = torch.abs(self.learnable_laplace(x))
        
        # 拼接特征学习权重
        features_cat = torch.cat([sobel_x, sobel_y, laplace], dim=1)
        weights = self.weight_net(features_cat)  # [B, 3, 1, 1]
        
        # 加权融合
        w1, w2, w3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        boundary_enhanced = w1 * sobel_x + w2 * sobel_y + w3 * laplace
        
        # 残差连接 + 输出转换
        output = x + boundary_enhanced  # 残差连接
        output = self.output_conv(output)
        
        return output

    def initialize(self):
        weight_init(self)

# 通道注意力模块：
class CA(nn.Module):
    def __init__(self, in_channel_left, in_channel_down):
        super(CA, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.bn0   = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_down, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True) #256
        down = down.mean(dim=(2,3), keepdim=True)#2,3指在H、W上取平均，保持维度不变，将每个通道的空间信息压缩成一个标量，后面两行代码最终生成了注意力权重。
        down = F.relu(self.conv1(down), inplace=True)
        down = torch.sigmoid(self.conv2(down))
        return left * down  #对应通道所有空间位置相乘（广播机制），实现通道注意力机制。

# 特征增强模块，扩大out1(256)到out2(512),然后再拆分，前256是w，后256是b，残差相加。
""" Self Refinement Module """
class SRM(nn.Module):
    def __init__(self, in_channel):
        super(SRM, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True) #256
        out2 = self.conv2(out1)
        w, b = out2[:, :256, :, :], out2[:, 256:, :, :]
        return F.relu(w * out1 + b, inplace=True)

""" Feature Interweaved Aggregation Module """
class FAM(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, in_channel_right):
        super(FAM, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_down, 256, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channel_right, 256, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(256)

        self.conv_d1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_d2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv_att1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.conv_att2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.conv_att3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, left, down, right):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True) #256 channels
        down = F.relu(self.bn1(self.conv1(down)), inplace=True) #256 channels
        right = F.relu(self.bn2(self.conv2(right)), inplace=True) #256

        down_1 = self.conv_d1(down)

        w1 = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear', align_corners=False)
            z1 = F.relu(w1 * down_, inplace=True)
        else:
            z1 = F.relu(w1 * down, inplace=True)
        z1_att = F.adaptive_avg_pool2d(self.conv_att1(z1), (1,1))
        z1 = z1_att * z1

        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode='bilinear', align_corners=False)

        z2 = F.relu(down_1 * left, inplace=True)
        z2_att = F.adaptive_avg_pool2d(self.conv_att2(z2), (1,1))
        z2 = z2_att * z2

        down_2 = self.conv_d2(right)
        if down_2.size()[2:] != left.size()[2:]:
            down_2 = F.interpolate(down_2, size=left.size()[2:], mode='bilinear', align_corners=False)
        z3 = F.relu(down_2 * left, inplace=True)
        z3_att = F.adaptive_avg_pool2d(self.conv_att3(z3), (1,1))
        z3 = z3_att * z3
        out = (z1 + z2 + z3) / (z1_att + z2_att + z3_att)# out = torch.cat((z1, z2, z3), dim=1)

        return F.relu(self.bn3(self.conv3(out)), inplace=True)

#自注意力模块： 对于in_channel_down这张512通道的特征图，被分成了256权重和256偏置，最终的通道数与in_channel_left一致是256
class SA(nn.Module):
    def __init__(self, in_channel_left, in_channel_down):
        super(SA, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channel_down, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True) #256 channels
        down_1 = self.conv2(down) #wb
        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode='bilinear', align_corners=False)
        w,b = down_1[:,:256,:,:], down_1[:,256:,:,:]

        return F.relu(w*left+b, inplace=True)
        
class HBNet(nn.Module):
    def __init__(self, cfg):
        super(HBNet, self).__init__()
        self.cfg     = cfg
        self.bkbone  = ResNet()
        self.apply(weight_init)  # 统一初始化

        # 前三个均来自于layer4,分别是Fh1,Fh2,Fh3
        self.ca45    = CA(2048, 2048)
        self.ca35    = CA(2048, 2048)
        self.ca25    = CA(2048, 2048)
        self.ca55    = CA(256, 2048)
        self.sa55   = SA(2048, 2048)

        self.fam45   = FAM(1024,  256, 256)
        self.fam34   = FAM( 512,  256, 256)
        self.fam23   = FAM( 256,  256, 256)

        self.srm5    = SRM(256)
        self.srm4    = SRM(256)
        self.srm3    = SRM(256)
        self.srm2    = SRM(256)

        self.linear5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.initialize()
         
        self.bem2 = BoundaryEnhancementModule(256,256)
        self.bem3 = BoundaryEnhancementModule(512,512)
        self.bem4 = BoundaryEnhancementModule(1024,1024)

        # 2类：边界/非边界
        self.boundary_head = nn.Conv2d(256, 2, kernel_size=1)
        
    def forward(self, x, mode=None,save_dir=None, iteration=0):
      
        out1, out2, out3, out4, out5_ = self.bkbone(x)
        
        out2 = self.bem2(out2)  
        out3 = self.bem3(out3)   
        out4 = self.bem4(out4)
   
        # GCF
        out4_a = self.ca45(out5_, out5_)
        out3_a = self.ca35(out5_, out5_)
        out2_a = self.ca25(out5_, out5_)
        # HA Fg 其中out5_a经过self.sa55(SA)变成了256
        out5_a = self.sa55(out5_, out5_)
        out5 = self.ca55(out5_a, out5_)

        # out 先进入解码器的layer1层再进行W操作
        out5 = self.srm5(out5)
        out4 = self.srm4(self.fam45(out4, out5, out4_a))   
        out3 = self.srm3(self.fam34(out3, out4, out3_a))       
        out2 = self.srm2(self.fam23(out2, out3, out2_a))    
        
        boundary_output = self.boundary_head(out2)
        # we use bilinear interpolation instead of transpose convolution
        if mode == 'Test':
            # ------------------------------------------------------ TEST ----------------------------------------------------
            out5 = F.interpolate(self.linear5(out5), size=x.size()[2:], mode='bilinear', align_corners=False)
            if save_dir is not None:
                save_heatmap(out5[0,0], save_path, "final_output5")
            out4 = F.interpolate(self.linear4(out4), size=x.size()[2:], mode='bilinear', align_corners=False)
            if save_dir is not None:
                save_heatmap(out4[0,0], save_path, "final_output4")
            out3 = F.interpolate(self.linear3(out3), size=x.size()[2:], mode='bilinear', align_corners=False)
            if save_dir is not None:
                save_heatmap(out3[0,0], save_path, "final_output3")
            out2 = F.interpolate(self.linear2(out2), size=x.size()[2:], mode='bilinear', align_corners=False)
            if save_dir is not None:
                save_heatmap(out2[0,0], save_path, "final_output2")
            return out2, out3, out4, out5, boundary_output
        else:
            # ------------------------------------------------------ TRAIN ----------------------------------------------------
            out2_no_sig  = F.interpolate(self.linear2(out2), size=x.size()[2:], mode='bilinear', align_corners=False)
            out5  = torch.sigmoid(F.interpolate(self.linear5(out5), size=x.size()[2:], mode='bilinear', align_corners=False))
            out4  = torch.sigmoid(F.interpolate(self.linear4(out4), size=x.size()[2:], mode='bilinear', align_corners=False))
            out3  = torch.sigmoid(F.interpolate(self.linear3(out3), size=x.size()[2:], mode='bilinear', align_corners=False))
            out2  = torch.sigmoid(out2_no_sig)
            
            out5 = torch.cat((1 - out5, out5), 1)
            out4 = torch.cat((1 - out4, out4), 1)
            out3 = torch.cat((1 - out3, out3), 1)
            out2 = torch.cat((1 - out2, out2), 1)
            
            return out2, out3, out4, out5,out2_no_sig, boundary_output
    def initialize(self):
        weight_init(self)
