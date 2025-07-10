#!/usr/bin/python3
#coding=utf-8
import os
import sys
#sys.path.insert(0, '../')
import torchvision
import time
sys.dont_write_bytecode = True
import cv2
import numpy as np
#import matplotlib.pyplot as plt
#plt.ion()
from thop import profile
from skimage import img_as_ubyte
import torch
import torch.nn.functional as F
import imageio
# from net_agg_bh_trans import SCWSSOD
# from net_agg_bh import SCWSSOD
# from net_agg import SCWSSOD
from HBNet import HBNet
from lib import dataset
from torch.utils.data import DataLoader
# import acc_iou

# folder_path = r"C:\Users\Ymk\Desktop\code\2-SCWSSOD-master\SCWSSOD-master\data\yzq\test\images"
# folder_path = r"C:\Users\Ymk\Desktop\dongtou_test\image"
folder_path = r'D:\github\liumengting\HBNet\data\test\image'
folder_path1 = r'D:\github\liumengting\HBNet\data\test'

file_list = os.listdir(folder_path)
# save_path = r'C:\Users\Ymk\Desktop\model_test\512\contrast\new_adapt\my_model_bh_out2'
save_path = r'D:\github\liumengting\test_out\ori_m_canny'
# pt_path = r"F:\model_ablation\my_model_bh_out2\my_model_bh_out2-80.pt"
# pt_path = r"D:\github\liumengting\HBNet\my_model_rw_bh-80.pt"
# pt_path = r"D:\github\liumengting\HBNet\model-80.pt"
pt_path = r"D:\github\liumengting\HBNet\hb_ori_m_canny80.pt"
# pt_path = r"D:\github\liumengting\HBNet\my_model_rw_bh1-80.pt"
# pt_path = r"D:\github\liumengting\HBNet\my_model_rw_bh2-80.pt"
# pt_path = r"D:\github\liumengting\HBNet\my_model_rw_bh3-80.pt"
# pt_path = r"C:\Users\Ymk\Desktop\code\3SCWSSOD-master\SCWSSOD-master\scwssod\loss0.4_scale0.5\loss0.4_scale0.5-50.pt"
for file_name in file_list:
    # 读取图像
    image_path = os.path.join(folder_path, file_name)
    print("file_name",file_name)
    print("folder_path",folder_path)
    image = cv2.imread(image_path) 
    if image is None:
        raise FileNotFoundError(f"Failed to load image at {image_path}")

    # # 确保图像成功读取后再转换
    image = image.astype(np.float32)[:, :, ::-1]  # BGR to RGB
#________________________预处理______________________________________________
    #归一化
    # mean = np.array([[[61.25,77.57,55.45]]])
    # std = np.array([[[4.74, 8.33,4.93]]])
    # 洞头数据集
    # mean = np.array([[[118.46049881,128.8552771,117.05277466]]])
    # std = np.array([[[31.33303169,31.62031719,30.20785496]]])
    #1000数据集
    # mean = np.array([[[61.59, 77.65, 55.42]]])
    # std = np.array([[[4.57, 7.61, 4.61]]])

    # test
    # mean = np.array([[[63.1129247,79.05180613,56.70662231]]])
    # std = np.array([[[5.2964298,9.07756342,5.28866402]]])

    #512_200(263)
    mean = np.array([[[109.561,124.17,101.23]]])
    std = np.array([[[24.26,30.46,22.46]]])

    #256_600
    # mean = np.array([[[60.27,75.75, 53.70]]])
    # std = np.array([[[4.10,5.48,3.71]]])

    # 1024_78
    # mean = np.array([[[79.16,91.76,72.67]]])
    # std = np.array([[[16.10,19.89,16.00]]])

    
    mean = np.array([[[0.485 * 256, 0.456 * 256, 0.406 * 256]]])
    std = np.array([[[0.229 * 256, 0.224 * 256, 0.225 * 256]]])
    image =  (image - mean) / std
    #Rizeze
    image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
    #ToTensor
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).unsqueeze(0)
#————————————————————————————————————————————————————————————————————————————————
    # 修改输入
    # folder_path = file_name.split("/")[-1]
    cfg = dataset.Config(datapath=folder_path1, mode='test')
    data   = dataset.Data(cfg)
    loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=8)
    ## network
    model  = HBNet(cfg)
    # 修改输入
    
    #读取模型
    # model = SCWSSOD(image.float())
    # model = HBNet(image.float())
    # model.load_state_dict(torch.load(pt_path, map_location=torch.device('cpu'), weights_only=True)) # weights_only 参数是安全加载权重 pytorch>=1.13.0
    model.load_state_dict(torch.load(pt_path, map_location=torch.device('cpu')))
    #不会进行反向传播和参数更新
    model.eval()

    macs, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),))  # 逗号使其成为元组
    print(f"参数量: {params / 1e6:.2f} M")  # 单位：百万
    print(f"计算量 (MACs): {macs / 1e9:.2f} G")  # 单位：GFLOPs

    start_time = time.time()

    #调整结果
    with torch.no_grad():  # 关闭梯度计算，加快推理速度
        out2, out3, out4, out5 = model(image.float(), 'Test')
        out2 = F.interpolate(out2, size=(512, 512), mode='bilinear', align_corners=False)
    end_time = time.time()

    inference_time = end_time - start_time
    print(f"Inference time for {file_name}: {inference_time:.4f} seconds")

    pred = (torch.sigmoid(out2[0, 0])).cpu().detach().numpy()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    pred = img_as_ubyte(pred)
    _, pred = cv2.threshold(img_as_ubyte(pred), 128, 255, cv2.THRESH_BINARY)

    imageio.imwrite(save_path + '/' + file_name, pred)



#评价指标
# true_dir = r'C:\Users\Ymk\Desktop\sx\paperimg\testimg\newDataset_clip256_true'
# mean_precision, mean_accuracy, mean_recall, mean_iou, mean_f1_score = acc_iou.batch_metrics(true_dir, save_path)
