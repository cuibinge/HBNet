#!/usr/bin/python3
#coding=utf-8

import os
import sys
#sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from skimage import img_as_ubyte
import torch
torch.backends.cudnn.enabled = False  # 禁用cuDNN
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
from lib import dataset
from HBNet import HBNet
import time
import logging as logger

TAG = "HBNet"
SAVE_PATH = TAG
GPU_ID=0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="test_%s.log"%(TAG), filemode="w")


DATASETS = ['./data/test',]


class Test(object):
    def save_probability_maps(self):
        """保存连续渐变的概率灰度图"""
        with torch.no_grad():
            # 灰度图保存
            # save_dir = r"D:\github\liumengting\HBNet\picture"
            # os.makedirs(save_dir, exist_ok=True)
            # 灰度图保存
            
            for image, mask, (H, W), name in self.loader:
                # 获取网络输出
                out2, out3, out4, out5 = self.net(image.cuda().float(), 'Test')
                
                # 调整大小并转换为概率
                out2 = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=False)
                prob_map = torch.sigmoid(out2[0, 0]).cpu().numpy()  # 获取概率图
                
                # 确保概率图是连续渐变的（关键修改）
                # plt.figure(figsize=(10, 10))
                # plt.imshow(prob_map, cmap='gray', vmin=0, vmax=1)  # 固定显示范围
                # plt.axis('off')
                
                # 保存为高质量灰度图
                # save_path = os.path.join(save_dir, name[0])
                # plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
                # plt.close()
                # print(f"Saved probability map to: {save_path}")
            
    def __init__(self, Dataset, datapath, Network):
        ## dataset
        self.datapath = datapath.split("/")[-1]
        print("Testing on %s"%self.datapath)
        self.cfg = Dataset.Config(datapath=datapath, mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=True, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        # self.net    = nn.DataParallel(self.net)
        path = './model-80.pt'
        #path = './my_model_rw_bh4-80.pt'
        state_dict = torch.load(path)
        print('complete loading: {}'.format(path))
        self.net.load_state_dict(state_dict)
        print('model has {} parameters in total'.format(sum(x.numel() for x in self.net.parameters())))
        self.net.train(False)
        self.net.cuda()
        self.net.eval()

    def accuracy(self):
        with torch.no_grad():
            mae, fscore, cnt, number   = 0, 0, 0, 256
            mean_pr, mean_re, threshod = 0, 0, np.linspace(0, 1, number, endpoint=False)
            cost_time = 0
            ##热力图---
            # for idx, (image, mask, (H, W), maskpath) in enumerate(self.loader):
            #     # 添加热力图保存参数
            #     # save_dir = "./heatmaps/1"  # 指定热力图保存根目录
            #     # save_dir = "./heatmap/1"  # 指定热力图保存根目录
            #     save_dir = "./heatmaps"  # 指定热力图保存根目录
            #     out2 = self.net(image.cuda().float(), 
            #                   'Test',
            #                   save_dir=save_dir,  # 传递保存路径
            #                   iteration=idx)      # 传递迭代次数
            ##热力图---
            for image, mask, (H, W), maskpath in self.loader:
                image, mask            = image.cuda().float(), mask.cuda().float()
                start_time = time.time()
                out2, out3, out4, out5 = self.net(image, 'Test')
                pred                   = torch.sigmoid(out2)
                torch.cuda.synchronize()
                end_time = time.time()
                cost_time += end_time - start_time

                ## MAE
                cnt += 1
                mae += (pred-mask).abs().mean()
                ## F-Score
                precision = torch.zeros(number)
                recall    = torch.zeros(number)
                for i in range(number):
                    temp         = (pred >= threshod[i]).float()
                    precision[i] = (temp*mask).sum()/(temp.sum()+1e-12)
                    recall[i]    = (temp*mask).sum()/(mask.sum()+1e-12)
                mean_pr += precision
                mean_re += recall
                fscore   = mean_pr*mean_re*(1+0.3)/(0.3*mean_pr+mean_re+1e-12)
                if cnt % 20 == 0:
                    fps = image.shape[0] / (end_time - start_time)
                    print('MAE=%.6f, F-score=%.6f, fps=%.4f'%(mae/cnt, fscore.max()/cnt, fps))
            fps = len(self.loader.dataset) / cost_time
            msg = '%s MAE=%.6f, F-score=%.6f, len(imgs)=%s, fps=%.4f'%(self.datapath, mae/cnt, fscore.max()/cnt, len(self.loader.dataset), fps)
            print(msg)
            logger.info(msg)

    def save(self):
        with torch.no_grad():
            for image, mask, (H, W), name in self.loader:
                out2, out3, out4, out5 = self.net(image.cuda().float(), 'Test')
                out2 = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=False)
                pred = (torch.sigmoid(out2[0, 0])).cpu().numpy()
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                head     = './pred_maps/{}/'.format(TAG) + self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0], img_as_ubyte(pred))


if __name__=='__main__':
    for e in DATASETS:
        t =Test(dataset, e, HBNet)
        t.accuracy()
        t.save()
        t.save_probability_maps()  # 新增调用