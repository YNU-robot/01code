#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# 导入OpenCV库
import cv2

# 导入PyTorch库
from torch import nn
import torch

# 导入自定义库
from models import AutoDriveNet
from utils import *


def main():
    '''
    主函数
    '''
    # 测试图像
    imgPath = './145_-0.04.jpg'

    # 推理环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载训练好的模型
    checkpoint = torch.load('./checkpoint2_color.pth')
    model = AutoDriveNet()
    model = model.to(device)
    model.load_state_dict(checkpoint['model'],strict=False)

    # 加载图像
    img = cv2.imread(imgPath)
    img = cv2.resize(img, (160,120))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 图像预处理
    # PIXEL_MEANS = (0.485, 0.456, 0.406)  # RGB格式的均值和方差
    # PIXEL_STDS = (0.229, 0.224, 0.225)
    img = torch.from_numpy(img.copy()).float()
    img /= 255.0
    # img -= torch.tensor(PIXEL_MEANS)
    # img /= torch.tensor(PIXEL_STDS)
    img = img.permute(2, 0, 1)
    img.unsqueeze_(0)

    # 转移数据至设备
    img = img.to(device)

    # 模型推理
    model.eval()
    with torch.no_grad():
        prelabel = model(img).squeeze(0).cpu().detach().numpy()
        print('预测结果  {:.3f} '.format(prelabel[0]))


if __name__ == '__main__':
    '''
    程序入口
    '''
    main()