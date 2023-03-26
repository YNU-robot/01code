# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import math

from numpy import angle
import torch
# 导入PyTorch库
from torch import nn

# 导入自定义库
from models import AutoDriveNet
from utils import *

def slope(x1, y1, x2, y2):
    """计算斜率

    Args:
        x1 (float): 第一个点的x坐标
        y1 (float): 第一个点的y坐标
        x2 (float): 第二个点的x坐标
        y2 (float): 第二个点的y坐标

    Returns:
        float: 斜率角的弧度
    """
    try:
        # 斜率k
        k = float(y2 - y1) / float(x2 - x1)
        theta = math.atan(k)
        res = theta * (  180 / math.pi)
    except ZeroDivisionError:
        print("垂直线")
        # 垂直线
        theta = 90
        res = theta * (  180 / math.pi)
    return res


def region_of_interest(edges, direction='left'):
    height, width = edges.shape
    mask = np.zeros_like(edges)
    # 定义感兴趣区域掩码轮廓，决定了进行识别的视野范围
    if direction == 'left':
        # 多边形的四个点
        polygon = np.array([[(0, height * 1 / 2),
                             (width * 1 / 2, height * 1 / 2),
                             (width * 1 / 2, height),
                             (0, height)]], np.int32)
    else:
        polygon = np.array([[(width * 1 / 2, height * 1 / 2),
                             (width, height * 1 / 2),
                             (width, height),
                             (width * 1 / 2, height)]], np.int32)
    # 填充感兴趣区域掩码
    cv2.fillPoly(mask, polygon, 255)
    # 提取感兴趣区域
    croped_edge = cv2.bitwise_and(edges, mask)
    return croped_edge


def detect_line(edges):
    '''
    基于霍夫变换的直线检测
    '''
    rho = 1  # 距离精度：1像素
    angle = np.pi / 180  # 角度精度：1度
    min_thr = 10  # 最少投票数
    lines = cv2.HoughLinesP(edges,
                            rho,
                            angle,
                            min_thr,
                            np.array([]),
                            minLineLength=8,
                            maxLineGap=8)
    return lines


def make_points(frame, line):
    '''
    根据直线斜率和截距计算线段起始坐标
    '''
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1 * 1 / 2)
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


def average_lines(frame, lines, direction='left'):
    # https://blog.csdn.net/yang332233/article/details/122120160
    '''
    小线段聚类
    '''
    lane_lines = []
    if lines is None:
        print(direction + '没有检测到线段')
        return lane_lines
    height, width, _ = frame.shape
    fits = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            # 计算拟合直线
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if direction == 'left' and slope < 0:
                fits.append((slope, intercept))
            elif direction == 'right' and slope > 0:
                fits.append((slope, intercept))
    if len(fits) > 0:
        fit_average = np.average(fits, axis=0)
        lane_lines.append(make_points(frame, fit_average))
    return lane_lines

def FitPolynomialCurve(img, n=5):
    '''
    拟合曲线
    '''
    h, w = img.shape[:2]
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    y, x = np.meshgrid(y, x)
    x = x.flatten()
    y = y.flatten()
    A = np.ones((len(x), n + 1))
    for i in range(1, n + 1):
        A[:, i] = x ** i
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    return coeffs


def display_line(frame, lines, line_color=(0, 0, 255), line_width=2):
    '''
    在原图上展示线段
    '''
    line_img = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), line_color, line_width)
    line_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    return line_img


if __name__ == "__main__":

    import time
    from Transbot_Lib import Transbot
    bot = Transbot()

    # 打开日志文件
    log_file = open('log.txt', 'w')

    cap = cv2.VideoCapture(0)

    a = b = c = 1


    # 推理环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载训练好的模型
    varpath = input('./results/checkpoint.pth:\n')
    checkpoint = torch.load(varpath)
    model = AutoDriveNet()
    model = model.to(device)
    model.load_state_dict(checkpoint['model'],strict=False)

    if cap.isOpened():
        window_handle = cv2.namedWindow("D435", cv2.WINDOW_AUTOSIZE)

        # 逐帧显示
        while cv2.getWindowProperty("D435", 0) >= 0:
            ret_val, img = cap.read()

            # height, width, _ = img.shape
            
            # 黄色hsv范围
            y_lower_hsv = np.array([23, 43, 46])
            y_upper_hsv = np.array([34, 255, 255])
            # 灰色hsv范围
            # g_lower_hsv = np.array([0, 0, 46])
            # g_upper_hsv = np.array([180, 43, 220])
            # 白色hsv范围
            w_lower_hsv = np.array([0, 0, 221])
            w_upper_hsv = np.array([180, 30, 255])

            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            y_mask = cv2.inRange(img2, y_lower_hsv, y_upper_hsv)
            w_mask = cv2.inRange(img2, w_lower_hsv, w_upper_hsv) 

            # 黄色部分识别降噪，使用开运算的方式，先腐蚀再膨胀，kernel的大小要根据情况适度调整，kernel越大修改的粒度就越大，最终结果是大块的黄色更容易保留
            kernel = np.ones((5, 5), np.uint8)
            y_mask = cv2.morphologyEx(y_mask, cv2.MORPH_OPEN, kernel)
            y_mask = cv2.morphologyEx(y_mask, cv2.MORPH_CLOSE, kernel)

            # 提取感兴趣区域
            y_roi = region_of_interest(y_mask)
            w_roi = region_of_interest(w_mask)

            # 将y_mask和w_mask合并
            mask = cv2.bitwise_or(y_roi, w_roi)
            print(type(mask))

            # 加载图像
            img = mask.copy()
            img = cv2.resize(img, (160,120))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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

             
            angle_to_mid_radian = prelabel[0]
 
            if abs(angle_to_mid_radian) > 0.5:
                if angle_to_mid_radian > 0:
                    angle_to_mid_radian = 0.5
                else:
                    angle_to_mid_radian = -0.5
                bot.set_car_motion(0, angle_to_mid_radian)
                print(f"0, {angle_to_mid_radian}")
                log_file.write(f"0, {angle_to_mid_radian}\n")
            else:
                # 执行动作，线速度为0.05， 转向角为angle_to_mid_radian
                bot.set_car_motion(0.05, angle_to_mid_radian)
                print(f"0.05, {angle_to_mid_radian}")
                log_file.write(f"0.05, {angle_to_mid_radian}\n")
            time.sleep(0.04)
            bot.set_car_motion(0.05, 0)
            


            cv2.imshow('D435', mask)


 

            # # 提取黄色部分
            # # 黄色的值范围
            # lower_hsv = np.array([26, 43, 46])
            # upper_hsv = np.array([34, 255, 255])

            # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # mask = cv2.inRange(img2, lower_hsv, upper_hsv)

            # # 黄色部分识别降噪，使用开运算的方式，先腐蚀再膨胀，kernel的大小要根据情况适度调整，kernel越大修改的粒度就越大，最终结果是大块的黄色更容易保留
            # kernel = np.ones((3, 3), np.uint8)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # # # 找到轮廓
            # # ret, contours, hierarchy = cv2.findContours(
            # #     opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # # # 绘制轮廓, 用红色
            # # cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

            # # canny边缘检测
            # yellow_edge = cv2.Canny(mask, 200, 400)

            # # 通过开运算提取除水平线，再减去水平线
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
            # yellow_edge2 = cv2.morphologyEx(yellow_edge, cv2.MORPH_OPEN, kernel)
            # yellow_edge2 = cv2.subtract(yellow_edge, yellow_edge2)

            # # 提取感兴趣区域
            # left_roi = region_of_interest(yellow_edge, 'left')
            # right_roi = region_of_interest(yellow_edge, 'right')
            # # 基于霍夫变换的直线检测
            # left_lines = detect_line(left_roi)
            # right_lines = detect_line(right_roi)

            # # 小线段聚类
            # left_lines = average_lines(img, left_lines, 'left')
            # right_lines = average_lines(img, right_lines, 'right')
            # # 在原图上展示线段
            # img = display_line(img, left_lines, (0, 0, 255), 2)
            # img = display_line(img, right_lines, (0, 0, 255), 2)

            # color_and_edge = np.hstack((yellow_edge, yellow_edge2))
            # # 将left_roi, right_roi合成一张图
            # roi = cv2.addWeighted(left_roi, 1, right_roi, 1, 1)

            # cv2.imshow("D435-color_edge", color_and_edge)
            # cv2.imshow("D435-roi", roi)
            # cv2.imshow("D435", img)

            # # 计算转向角
            # x_offset = 0
            # y_offset = 0
            # if len(left_lines) > 0 and len(right_lines) > 0:  # 检测到2条线
            #     _, _, left_x2, _ = left_lines[0][0]
            #     _, _, right_x2, _ = right_lines[0][0]
            #     mid = int(width / 2)
            #     x_offset = (left_x2 + right_x2) / 2 - mid
            #     y_offset = int(height / 2)
            # elif len(left_lines) > 0 and len(left_lines[0]) == 1:  # 只检测到左行道线
            #     print("只检测到左行道线")
            #     log_file.write("只检测到左行道线\n")
            #     x1, _, x2, _ = left_lines[0][0]
            #     x_offset = x2 - x1
            #     y_offset = int(height / 2)
            # elif len(right_lines) > 0 and len(right_lines[0]) == 1:  # 只检测到右行道线
            #     print("只检测到右行道线")
            #     log_file.write("只检测到右行道线\n")
            #     x1, _, x2, _ = right_lines[0][0]
            #     x_offset = x2 - x1
            #     y_offset = int(height / 2)
            # else:  # 一条线都没检测到
            #     print('检测不到行道线')
            #     log_file.write('检测不到行道线\n')
            #     bot.set_car_motion(0, 0)
            #     time.sleep(1)
            #     # break
            #     continue
            
            # angle_to_mid_radian = -math.atan(x_offset / y_offset)  
            # angle_to_mid_deg = -int(angle_to_mid_radian * 180.0 / math.pi) 
            # # 转向角??
            # # steering_angle = angle_to_mid_deg/45.0
 
            # # if abs(angle_to_mid_radian) > 0.5:
            # #     if angle_to_mid_radian > 0:
            # #         angle_to_mid_radian = 0.5
            # #     else:
            # #         angle_to_mid_radian = -0.5
            # #     bot.set_car_motion(0, angle_to_mid_radian)
            # #     print(f"0, {angle_to_mid_radian}")
            # #     log_file.write(f"0, {angle_to_mid_radian}\n")
            # # else:
            # #     # 执行动作，线速度为0.05， 转向角为angle_to_mid_radian
            # #     bot.set_car_motion(0.05, angle_to_mid_radian)
            # #     print(f"0.05, {angle_to_mid_radian}")
            # #     log_file.write(f"0.05, {angle_to_mid_radian}\n")
            # # time.sleep(0.04)
            # # bot.set_car_motion(0.05, 0)
            


            
            keyCode = cv2.waitKey(30) & 0xFF
            if keyCode == 27:  # ESC键退出
                break
        
        cap.release()
        cv2.destroyAllWindows()

    else:
        print("打开摄像头失败")

    
    del bot
    log_file.close()
