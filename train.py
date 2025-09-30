#!/usr/bin/python3
#coding=utf-8

import sys
import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data import dataset
from net_agg_bp import SCWSSOD
import logging as logger
from lib.data_prefetcher import DataPrefetcher
from lscloss import *
import numpy as np
from tools import *
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte, img_as_float
# 新增导入
from DCBP_training import Focal_Loss, CE_Loss, Dice_loss

TAG = "scwssod"
SAVE_PATH = "semi_train/general_21_1100_bp_bl0.15"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="train_%s.log"%(TAG), filemode="w")


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def generate_boundary_from_mask(mask, dilation_radius=1):
    """
    从分割掩码生成边界标签
    mask: [batch, height, width] 分割标签
    返回: [batch, height, width] 边界标签 (0:非边界, 1:边界)
    """
    boundary_masks = []
    mask_np = mask.cpu().numpy().astype(np.uint8)
    
    for i in range(mask_np.shape[0]):
        single_mask = mask_np[i]
        boundaries = np.zeros_like(single_mask)
        unique_labels = np.unique(single_mask)
        
        for label in unique_labels:
            if label == 0 or label == 255:  # 忽略背景和ignore_index
                continue
                
            # 创建当前物体的二值掩码
            obj_mask = (single_mask == label).astype(np.uint8)
            
            # 使用形态学操作找到边界
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(obj_mask, kernel, iterations=dilation_radius)
            eroded = cv2.erode(obj_mask, kernel, iterations=dilation_radius)
            boundary = dilated - eroded
            
            boundaries[boundary > 0] = 1
        
        boundary_masks.append(boundaries)
    
    return torch.from_numpy(np.stack(boundary_masks)).long().cuda()

""" set lr """
def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
        annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps*ratio)
    last  = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur/total_steps)
    x = np.abs(cur*2.0/total_steps - 2.0*cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr)*cur + min_lr*first - base_lr*total_steps)/(first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1.-x)
        else:
            momentum = momentums[0]

    return lr, momentum


def get_polylr(base_lr, last_epoch, num_steps, power):
    return base_lr * (1.0 - min(last_epoch, num_steps-1) / num_steps) **power


BASE_LR = 1e-5
MAX_LR = 1e-2
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
batch = 8
l = 0.3

def train(Dataset, Network):
    ## dataset
    cfg = Dataset.Config(datapath='/wangfaqi/data/paper_data_add_dt100/train', savepath=SAVE_PATH, mode='train', batch=batch, lr=1e-3, momen=0.9, decay=5e-4,scale = 0.5, epoch=90)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8)
    ## network
    net = Network(cfg)
    # print('model has {} parameters in total'.format(sum(x.numel() for x in net.parameters())))
    criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean')
    loss_lsc = LocalSaliencyCoherence().cuda()
    net.train(True)
    net.cuda()
    criterion.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    sw = SummaryWriter(cfg.savepath)
    global_step = 0

    db_size = len(loader)

    # -------------------------- training ------------------------------------
    for epoch in range(cfg.epoch):
        prefetcher = DataPrefetcher(loader)
        batch_idx = -1
        image, mask = prefetcher.next()
        while image is not None:
            niter = epoch * db_size + batch_idx
            lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, cfg.epoch*db_size, niter, ratio=1.)
            optimizer.param_groups[0]['lr'] = 0.1 * lr  # for backbone
            optimizer.param_groups[1]['lr'] = lr
            optimizer.momentum = momentum
            batch_idx += 1
            global_step += 1

            ######  saliency structure consistency loss  ######
            ###########################   scale  ######################################
            image_scale = F.interpolate(image, scale_factor = cfg.scale, mode='bilinear', align_corners=True)         
            out2, out3, out4, out5,out2_no_sig, boundary_output = net(image, 'Train')
            out2_s, out3_s, out4_s, out5_s,_,_ = net(image_scale, 'Train')
            out2_scale = F.interpolate(out2[:, 1:2], scale_factor = cfg.scale, mode='bilinear', align_corners=True)
            loss_ssc = SaliencyStructureConsistency(out2_s[:, 1:2], out2_scale, 0.85)
            
            ###### 边界检测Focal Loss (新增) ######
            # 从分割标签生成边界标签
            boundary_gt = generate_boundary_from_mask(mask.squeeze(1))
            boundary_loss = Focal_Loss(boundary_output, boundary_gt, cls_weights=None, 
                                      num_classes=2, alpha=0.25, gamma=2.0)
            
            
 #-----------------jx-loss---------------------------------------------------
            _, _, _, _ ,out2_no_sig, _= net(image.float(), 'Train')
            #out2_no_sig = F.interpolate(out2_no_sig[:, 1:2], size=(256, 256), mode='bilinear', align_corners=False)
            out2_no_sig = (torch.sigmoid(out2_no_sig[0, 0])).cpu().detach().numpy()
            out2_no_sig = (out2_no_sig - out2_no_sig.min()) / (out2_no_sig.max() - out2_no_sig.min() + 1e-8)
            fake_truth = img_as_ubyte(out2_no_sig)

            # 矩形拟合 ： 边缘检测
            edges = cv2.Canny(fake_truth, 50, 150)
            # 提取轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 创建一个与二值图像相同大小的空图像
            filled_image = np.zeros_like(fake_truth)

            # 进行矩形拟合和填充
            for contour in contours:
                # 进行矩形拟合
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # 填充矩形区域
                cv2.fillPoly(filled_image, [box], color=(255, 255, 255))

            filled_image = img_as_float(filled_image)
            loss_fn1 = torch.nn.MSELoss()
            loss6 = loss_fn1(torch.from_numpy(out2_no_sig).float(), torch.from_numpy(filled_image).float())
#----------------------juxing--------------------------------------------------------         

            ######   label for partial cross-entropy loss  ######
            gt = mask.squeeze(1).long()
            bg_label = gt.clone()
            fg_label = gt.clone()
            bg_label[gt != 0] = 255
            fg_label[gt == 0] = 255

            ######   local saliency coherence loss (scale to realize large batchsize)  ######
            image_ = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=True)
            sample = {'rgb': image_}
            out2_ = F.interpolate(out2[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
            loss2_lsc = loss_lsc(out2_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
            loss2 = loss_ssc + criterion(out2, fg_label) + criterion(out2, bg_label) + l * loss2_lsc  ## dominant loss

            ######  auxiliary losses  ######
            out3_ = F.interpolate(out3[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
            loss3_lsc = loss_lsc(out3_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
            loss3 = criterion(out3, fg_label) + criterion(out3, bg_label) + l * loss3_lsc
            out4_ = F.interpolate(out4[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
            loss4_lsc = loss_lsc(out4_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
            loss4 = criterion(out4, fg_label) + criterion(out4, bg_label) + l * loss4_lsc
            out5_ = F.interpolate(out5[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
            loss5_lsc = loss_lsc(out5_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
            loss5 = criterion(out5, fg_label) + criterion(out5, bg_label) + l * loss5_lsc

            ######  objective function  ######
            #loss6 = loss5
            loss = loss2*1 + loss3*0.8 + loss4*0.6 + loss5*0.4 + loss6*0.4 + boundary_loss*0.15
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            if batch_idx % 10 == 0:
                msg = '%s| %s | step:%d/%d/%d | lr=%.6f | loss=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f | loss5=%.6f| loss6=%.6f' % (SAVE_PATH, datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(),loss6.item())
                print(msg)
                logger.info(msg)
            image, mask = prefetcher.next()
        if epoch > 28:
            if (epoch+1) % 10 == 0 or (epoch+1) == cfg.epoch:
                torch.save(net.state_dict(), cfg.savepath+'/semi_new-'+str(epoch+1)+'.pt')

if __name__=='__main__':
    train(dataset, SCWSSOD)
