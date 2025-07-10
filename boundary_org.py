import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import PIL
def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for m in model.modules():
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
def Upsample(x, size):
    return nn.functional.interpolate(x, size, mode='bilinear', align_corners=True)
class ResBlock(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1):
        super(ResBlock, self).__init__()
        ## conv branch
        self.left = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chs)
        )
        ## shortcut branch
        self.short_cut = nn.Sequential()
        if stride != 1 or in_chs != out_chs:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chs))

    ### get the residual
    def forward(self, x):
        return F.relu(self.left(x) + self.short_cut(x))
class ModuleHelper:
    @staticmethod
    def BNReLU(num_features, inplace=True):
        return nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=inplace)
        )

    @staticmethod
    def BatchNorm2d(num_features):
        return nn.BatchNorm2d(num_features)

    @staticmethod
    def Conv3x3_BNReLU(in_channels, out_channels, stride=1, dilation=1, groups=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                      groups=groups, bias=False),
            ModuleHelper.BNReLU(out_channels)
        )

    @staticmethod
    def Conv1x1_BNReLU(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            ModuleHelper.BNReLU(out_channels)
        )

    @staticmethod
    def Conv1x1(in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)
class Conv1x1(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(Conv1x1, self).__init__()

        self.conv = ModuleHelper.Conv1x1_BNReLU(in_chs, out_chs)

        initialize_weights(self.conv)

    def forward(self, x):
        return self.conv(x)

 #将任意维度的特征转化为通道数为3的特征
class ConvTo3Channels(nn.Module):
    def __init__(self, in_channels):
        super(ConvTo3Channels, self).__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, x):
        out = self.conv(x)
        return out
class BoundaryEnhancementModule(nn.Module):
    def __init__(self, in_chs=3, out_chs=128):#更改通道数，以求对应
        
        super(BoundaryEnhancementModule, self).__init__()
        self.horizontal_conv = nn.Sequential(
            nn.Conv2d(in_chs, 128, (1, 7)),
            ModuleHelper.BNReLU(128),
#             ResBlock(128, 128),
            
            Conv1x1(128, 1)
        )  # bs,1,352,346
        self.conv1x1_h = Conv1x1(2, 8)

        self.vertical_conv = nn.Sequential(
            nn.Conv2d(in_chs, 128, (7, 1)),
            ModuleHelper.BNReLU(128),
#             ResBlock(128, 128),
            
            Conv1x1(128, 1)
            
        )
        
        self.conv1x1_v = Conv1x1(2, 8)
        self.conv_out = Conv1x1(16, out_chs)

        self.transto3 = ConvTo3Channels(in_chs)
    def forward(self, x, save_path="D:/github/liumengting/HBNet/heatmaps/sample"):
        import os
        import random
        rand_num = random.randint(1000, 9999)  # 4位随机数
        save_path = os.path.join(save_path, f"sample_{rand_num}")
        os.makedirs(save_path, exist_ok=True)
        #print('传进来的是'+ str(x.shape))
        #x:ou2:16,256,80,80
        bs, chl, w, h = x.size()[0], x.size()[1], x.size()[2], x.size()[3]
        x_h = self.horizontal_conv(x)
        x_h = Upsample(x_h, (w, h))
        x_v = self.vertical_conv(x)
        x_v = Upsample(x_v, (w, h))
        # x_arr = (x.cpu().detach().numpy().transpose((0, 2, 3, 1))*255).astype(np.uint8)
        x_arr = x.cpu().detach().numpy()
        canny = np.zeros((bs, 1, w, h))

        for i in range(bs):
            x_arr[x_arr > 1] /= 10
            ## print(f"x_arr[{i}] shape: {x_arr[i].shape}")
            img_normalized = (x_arr[i] - np.min(x_arr[i])) / (np.max(x_arr[i]) - np.min(x_arr[i]) + 1e-8) * 255
            img_normalized = img_normalized.astype(np.uint8)  # 转换为 uint8 类型
            ## print(f"Sample {i}:")
            ## 打印前5行和前5列的像素值（假设是单通道图像）
            ## print(x_arr[i, :80, :80, 0])  # 如果是多通道，可以调整通道索引
            np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            ## print("min and max",np.min(x_arr[i]), np.max(x_arr[i]))
            ## print(img_normalized[:80, :80, 0] if x_arr.ndim > 2 else x_arr[:80, :80])
            np.set_printoptions(threshold=1000, linewidth=75)
            ## 应用高斯模糊（可选）
            blurred = cv2.GaussianBlur(img_normalized[i], (5, 5), 0)
            canny_edges = cv2.Canny(blurred, 10, 100)
            ## canny_edges = cv2.Canny(x_arr[i], 100, 120)
            ## cv2.imshow('Canny Edges', canny_edges)
            ## cv2.waitKey(0)  # 等待按键以关闭窗口
            ## cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
            canny[i]=canny_edges
            ##canny[i] = cv2.Canny(x_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cpu().float().to(x_h.device)
        
        # Save heatmaps
        self.save_heatmap(x_h, "horizontal", save_path)
        self.save_heatmap(x_v, "vertical", save_path)
        self.save_canny(canny, "canny", save_path)

        h_canny = torch.cat((x_h, canny), dim=1)
        v_canny = torch.cat((x_v, canny), dim=1)
        h_v_canny = torch.cat((self.conv1x1_h(h_canny), self.conv1x1_v(v_canny)), dim=1)
        h_v_canny_out = self.conv_out(h_v_canny)

        return h_v_canny_out
    def save_heatmap(self, tensor, name_prefix, save_path):
        import os
        import matplotlib.pyplot as plt
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(tensor.size(0)):  # loop over batch
            
            
            # Convert tensor to numpy and squeeze single-channel dim if needed
            heatmap = tensor[i].squeeze().cpu().detach().numpy()
            
            # Normalize to [0, 1]
            if heatmap.max() != heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Apply colormap (blue-like)
            cmap = plt.cm.Blues
            colored_heatmap = cmap(heatmap)[..., :3]  # take RGB, ignore alpha
            colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
            
            # Save image
            filename = f"{save_path}/{name_prefix}_heatmap_{i}.png"
            cv2.imwrite(filename, cv2.cvtColor(colored_heatmap, cv2.COLOR_RGB2BGR))
    def save_canny(self, tensor, name_prefix, save_path):
        # import os
        # for i in range(tensor.size(0)):
        #     canny = tensor[i].squeeze().cpu().detach().numpy()
        #     canny = (canny * 255).astype(np.uint8)  # 直接保存二值图
        #     filename = f"{save_path}/{name_prefix}_canny_{i}.png"
        #     cv2.imwrite(filename, canny)  # 保存为黑白图像
        import os
        from PIL import Image
        for i in range(tensor.size(0)):
            canny = tensor[i].squeeze().cpu().detach().numpy()
            canny = (canny * 255).astype(np.uint8)  # 转换为 uint8 类型
            
            # 使用 PIL 保存图像
            pil_image = Image.fromarray(canny)
            filename = f"{save_path}/{name_prefix}_canny_{i}.png"
            pil_image.save(filename)
def initialize(self):
        initialize_weights(self)

def main():
    # cfg = Dataset.Config(datapath='./data/my_data_large', savepath=SAVE_PATH, mode='train', batch=batch, lr=1e-3,momen=0.9, decay=5e-4, epoch=50)
    # 随机生成与数据大小相同的数组
    random_array = np.random.rand(2, 3, 64, 64)
    #print(random_array)
    # 将随机数组转换为 PyTorch 的 Tensor 格式
    input_tensor = torch.Tensor(random_array)

    model = BoundaryEnhancementModule()
    #out2, out3, out4, out5 = model(input_tensor, 'Test')
    # 设置模型为评估模式
    model.eval()
    # Set save path
    save_path = "D:/github/liumengting/HBNet/heatmaps/sample"
    # 使用模型进行推理
    with torch.no_grad():
        output = model(input_tensor, save_path)
        # print(output)
        # print(output.shape)

if __name__ == '__main__':
    main()