#!/usr/bin/env python
# coding: utf-8

# In[31]:
import cv2
import numpy as np
import math

def c_slope(x1, y1, x2, y2):
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
        theta_rad = math.atan(k)
        # res = theta * (180 / math.pi) # 弧度转角度
    except ZeroDivisionError:
        # 垂直线
        theta_rad = math.pi / 2
        # res = theta * (180 / math.pi)
    return theta_rad


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
    # 合法性检查
    if slope == 0:
        slope = 0.0001
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

def average_lines_old(frame, lines, direction='middle'):
    # https://blog.csdn.net/yang332233/article/details/122120160
    '''
    小线段聚类
    direction: left or right or middle
    '''
    lane_lines = []
    if lines is None:
        print('没有检测到线段')
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
            elif direction == 'middle':
                fits.append((slope, intercept))
    if len(fits) > 0:
        fit_average = np.average(fits, axis=0)
        lane_lines.append(make_points(frame, fit_average))
    return lane_lines

def average_lines(frame, lines, direction='left'):
    # https://blog.csdn.net/yang332233/article/details/122120160
    '''
    小线段聚类, 聚类时忽略掉斜率在tan(0) - tan(pi/6)和tan(pi * 5/ 6) - tan(pi)之间的线段
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
            k = c_slope(x1, y1, x2, y2)
            # 丢弃斜率在tan(0) - tan(pi/6)或tan(pi * 5/ 6) - tan(pi)之间的线段
            if (k > math.tan(0) and k < math.tan(math.pi / 2)) or (k > math.tan(math.pi * 3 / 4) and k < math.tan(math.pi)):
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


# In[32]:

def show(window_name, img):
    def mouse_click(event, x, y, flags, para):
        if event == cv2.EVENT_LBUTTONDOWN:  # 左边鼠标点击
            print('PIX(y,x):', y, x)
            print("BGR:", img[y, x])

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click)
    cv2.imshow(window_name, img)
    if cv2.waitKey(0) == ord("q"):
        cv2.destroyAllWindows()


def scanLR(src: np.ndarray, step=1, visual=False):
    """扫线检测左右车道线
    因为对图像的最底部做扫线的时候默认使用终点，因此其值不准，后面拟合线或者判断的时候应该忽略底部的行（从img.shape[0] - 1 - step开始）

    Args:
        mask (np.ndarray): 输入的二值化图像
        step (int, optional): 扫描步长. Defaults to 1.

    """
    visual_img = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
    middle_line_mask = np.zeros((src.shape[0], src.shape[1]), dtype=np.uint8)
    # 提取黄色部分
    # 黄色的值范围
    lower_hsv = np.array([23, 43, 46])
    upper_hsv = np.array([34, 255, 255])

    img2 = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img2, lower_hsv, upper_hsv)

    # 黄色部分识别降噪，使用开运算的方式，先腐蚀再膨胀，kernel的大小要根据情况适度调整，kernel越大修改的粒度就越大，最终结果是大块的黄色更容易保留
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # middle[y] 表示第y行道路的中点坐标
    middle = np.zeros(mask.shape[0], dtype=np.int32)
    left = np.zeros(mask.shape[0], dtype=np.int32)
    right = np.zeros(mask.shape[0], dtype=np.int32)
    # 初始化第一行
    for i in range(mask.shape[0] - 1, mask.shape[0] - 1 - step + 1, -1):
        middle[i] = mask.shape[1] // 2
        right[i] = mask.shape[1] - 1
        left[i] = 0
    
    for y in range(mask.shape[0] - 1 - step, 0, -step):
        if y < (0 + step):
            break
        # 当前行标为y，上一行标为 y + step
        # 扫描右边
        for x in range(middle[y + step], mask.shape[1], 1):
            # 找到了右边的车道线，黑白白的方式
            if (x == mask.shape[1] - 1) or (x == mask.shape[1] - 2) or (mask[y][x] == 0 and mask[y][x + 1] == 255 and mask[y][x + 2] == 255):
                # right[y] = x
                right[y:y+step] = x
                break
        # 扫描左边
        for x in range(middle[y + step], -1, -1):
            # 找到了左边的车道线，黑白白的方式，前面的or表达式是为了防止后面下标记的越界
            if (x == -1 + 1) or (x == -1 + 1 + 1) or (mask[y][x] == 0 and mask[y][x - 1] == 255 and mask[y][x - 2] == 255):
                # left[y] = x
                left[y:y+step] = x
                break
        # 计算step中所有行的中点
        middle[y:y+step] = (left[y] + right[y]) //2

        # middle[y] = (left[y] + right[y]) // 2
        if visual:
            # visual_img[y][left[y]] = [0, 0, 255]
            # visual_img[y][right[y]] = [255, 0, 0]
            # visual_img[y][middle[y]] = [0, 255, 0]

            # --- 用以下切片方式会报错 ？？ 
            # print(y)
            # visual_img[y:y+step][left[y]] = [0,0,255]
            # visual_img[y:y+step][right[y]] = [255, 0, 0]
            # visual_img[y:y+step][middle[y]] = [0, 0, 255]
            # for i in range(step):
            #     visual_img[y+i][left[y]] = [0,0,255]
            #     visual_img[y+i][right[y]] = [255, 0,0]
            #     visual_img[y+i][middle[y]] = [0,255,0]
            #     middle_line_mask[y+i][middle[y]] = 255
            for i in range(step):
                visual_img[y:y+step, left[y], :] = [0,0,255]
                visual_img[y:y+step, right[y], :] = [255, 0,0]
                visual_img[y:y+step, middle[y], :] = [0,255,0]
                middle_line_mask[y:y+step, middle[y]] = 255
   
    return left, right, middle, visual_img, middle_line_mask

def road_type_detection(left: list, right: list, middle: list, visual_img: np.ndarray, visual=False):
    """判断道路类型

    Args:
        left (左): _description_
        right (右): _description_
        middle (中): _description_
        visual_img (np.ndarray): 三线图

    Returns:
        str: "straight" or "curve"
    """
    sum1: float = 0
    sum2: float = 0

    x_middle = visual_img.shape[1] // 2
    y_middle = visual_img.shape[0] // 2

    for i in range(y_middle, visual_img.shape[0]):
        sum1 += (x_middle - middle[i])
        sum2 += (middle[i] - x_middle) ** 2

    print("sum1:", sum1 / y_middle)
    print("sum2:", sum2 / y_middle)

    road_type = ""
    # 方差判断直道和弯道，3.2和18.5是经验值
    if (sum2 > (3.2 * y_middle) ** 2 or sum1 / y_middle > 18.5 or sum1 / y_middle < -18.5):
        road_type = "curve"
    else:
        road_type = "straight"
    if visual: 
        road_type = road_type_detection(left, right, middle, visual_img)
        # 在图片左上角显示文字
        visual_img = cv2.putText(visual_img, road_type, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
    return road_type

# In[27]:


# 检查拐点
def detect_turn_point(img:np.ndarray, left: list, right: list, middle: list,  visual_img: np.ndarray,step: int = 5, roi: float = 0.5, visual=False):
    """深蓝色表示右下，浅蓝色表示右上，红色表示左下，粉色左上

    Args:
        img (np.ndarray): 图片
        left (list): 左边界点
        right (list): 有边界点
        middle (list): 中点
        visual_img (np.ndarray): 可是画的标记点图
        step (int, optional): 扫描隔行数. Defaults to 5.
        roi (float, optional): 处理区域. Defaults to 0.5.
        visual (bool, optional): 是否可视化. Defaults to False.
    """

    lower_left_point_finded = False
    lower_right_point_finded = False
    upper_left_point_finded = False
    upper_right_point_finded = False
    lower_left_point = None
    lower_right_point = None
    upper_left_point = None
    upper_right_point = None

    # 需要注意左边0或者1表示丢失左边线，右边shape[1] - 1或者shape[1] - 2表示丢失右边线，这是因为scanLR函数中的left[i] != 0和right[i] != img.shape[1] - 1的判断条件
    for i in range(img.shape[0] - 1, math.floor(img.shape[0] * (1 - roi)), -step):
        if lower_left_point_finded == False:
            # 左下角未丢线
            # if left[img.shape[0] - 1] != 0 and left[img.shape[0] - 1 - step] != 0:
            # 丢线就跳过
            if left[i] != 0 and left[i] != 1:
                # 关于判定边界点跳变，一般采用经验值：1.5% - 5%，即两行之间的边界点横坐标差距为图像水平宽度的1.5% - 5%。该值越小对于跳变就越敏感，越大对于跳变就越不敏感。这里判断左边界突然左移。
                if (left[i - step] - left[i]) / img.shape[1] < -0.015:
                    p_x = np.max(left[i: img.shape[0]])
                    p_y = np.argmax(left[i: img.shape[0]])  # 查找最大值的索引
                    # 将索引转换为原图像的索引
                    p_y = p_y + i
                    lower_left_point = (p_y, p_x)
                    print("lower_left_point:", lower_left_point)
                    lower_left_point_finded = True

        # 右下角未丢线
        # if right[img.shape[0] - 1] != img.shape[1] and right[img.shape[0] - 1 - step] != img.shape[1]:
        if lower_right_point_finded == False:
            # 丢线就跳过
            if right[i] != img.shape[1] - 1 and right[i] != img.shape[1] - 2:
                # 关于判定边界点跳变，一般采用经验值：1.5% - 5%，即两行之间的边界点横坐标差距为图像水平宽度的1.5% - 5%。该值越小对于跳变就越敏感，越大对于跳变就越不敏感，这里判断突然增大
                if (right[i - step] - right[i]) / img.shape[1] > 0.015:
                    p_x = np.min(right[i: img.shape[0]])
                    p_y = np.argmin(right[i: img.shape[0]])  # 查找最小值的索引
                    # 将索引转换为原图像的索引
                    p_y = p_y + i
                    lower_right_point = (p_y, p_x)
                    print("lower_right_point:", lower_right_point)
                    lower_right_point_finded = True

        if upper_left_point_finded == False:
            # 关于判定边界点跳变，一般认为从丢线突变到有线，是一个较大的跳变，这里判断本行左边界本来丢线，突然出现，这里用经验值，放置丢线行的小干扰
            if (left[i] == 0 or left[i] == 1) and left[i - step] / img.shape[1] > 0.20:
                upper_left_point = (i - step, left[i - step])
                print("upper_left_point:", upper_left_point)
                upper_left_point_finded = True

        if upper_right_point_finded == False:
            if right[i] == img.shape[1] - 1:
                print(right[i], right[i - step])
            if (right[i] == img.shape[1] - 1 or right[i] == img.shape[1] - 2) and (img.shape[1] - 1 - right[i - step]) / img.shape[1] > 0.20:
                upper_right_point = (i - step, right[i - step])
                print("upper_right_point:", upper_right_point)
                upper_right_point_finded = True

        if lower_left_point_finded and lower_right_point_finded and upper_left_point_finded and upper_right_point_finded:
            # 标记拐点
            break
    # T形路口左补线
    if upper_left_point_finded == True  and lower_left_point_finded == True :
        y_start = lower_left_point[0]
        y_end = upper_left_point[0]
        x_start = lower_left_point[1]
        x_end = upper_left_point[1]
        slope = (x_end - x_start) / (y_end - y_start)
        for i in range(y_start, y_end, -1):
            if visual:
                visual_img[i, left[i]] = [0, 0, 0] # 删除原来的可视化点
                visual_img[i, middle[i]] = [0, 0, 0] # 删除原来的可视化点
            # 更新边界点和中线点
            left[i] = int(x_start + slope * (i - y_start))
            if left[i] < 0:
                left[i] = 0
            if right[i] == img.shape[1] - 1 or right[i] == img.shape[1] - 2:
                if visual:
                    visual_img[i, right[i]] = [0, 0, 0] # 删除原来的可视化点
                right[i] = 2*middle[i] - left[i]
                if right[i] > img.shape[1] - 1:
                    right[i] = img.shape[1] - 1
                if visual:
                    visual_img[i, right[i]] = [255,0, 0] # 添加新的可视化点
            middle[i] = int((left[i] + right[i]) / 2)
            if visual:
                visual_img[i, left[i]] = [0, 0, 255] # 添加新的可视化点
                visual_img[i, middle[i]] = [0, 255, 0] # 添加新的可视化点
    # T形路口右补线
    if upper_right_point_finded == True and lower_right_point_finded == True :
        y_start = lower_right_point[0]
        y_end = upper_right_point[0]
        x_start = lower_right_point[1]
        x_end = upper_right_point[1]
        slope = (x_end - x_start) / (y_end - y_start)
        for i in range(y_start, y_end, -1):
            if visual:
                visual_img[i, right[i]] = [0, 0, 0] # 删除原来的可视化点
                visual_img[i, middle[i]] = [0, 0, 0] # 删除原来的可视化点
            # 更新边界点和中线点
            right[i] = int(x_start + slope * (i - y_start)) 
            if right[i] > img.shape[1] - 1:
                right[i] = img.shape[1] - 1
            if left[i] == 0 or left[i] == 1:
                if visual:
                    visual_img[i, left[i]] = [0, 0, 0] # 删除原来的可视化点
                left[i] = 2 * middle[i + 1] - right[i] # 用上一行的中线点来补线
                if left[i] < 0:
                    left[i] = 0
                if visual:
                    visual_img[i, left[i]] = [0, 0, 255] # 添加新的可视化点
            middle[i] = int((left[i] + right[i]) / 2)
            if visual:
                visual_img[i, right[i]] = [255, 0, 0] # 添加新的可视化点
                visual_img[i, middle[i]] = [0, 255, 0] # 添加新的可视化点


    if visual:
        # 记录的拐点坐标是用的y,x，所以要[::-1]反转一下y和x的顺序
        if lower_left_point_finded == True:
            visual_img = cv2.circle(
                visual_img, lower_left_point[::-1], 5, (50, 0, 255), -1)
        if lower_right_point_finded == True:
            visual_img = cv2.circle(
                visual_img, lower_right_point[::-1], 5, (255, 50, 0), -1)
        if upper_left_point_finded == True:
            visual_img = cv2.circle(
                visual_img, upper_left_point[::-1], 5, (255, 0, 255), -1)
        if upper_right_point_finded == True:
            visual_img = cv2.circle(
                visual_img, upper_right_point[::-1], 5, (255, 255, 0), -1)

if __name__ == "__main__":

    import time
    from Transbot_Lib import Transbot
    bot = Transbot()

    # 打开日志文件
    log_file = open('log.txt', 'w')

    cap = cv2.VideoCapture(0)

    a = b = c = 1

    if cap.isOpened():
        window_handle = cv2.namedWindow("main", cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty("main", 0) >= 0:
            ret_val, img = cap.read()
            if ret_val == False:
                break
            # ------
            # todo 影响识别速度的参数：k分辨率压缩，scanLR step参数（取样横线数）
            # 将图片分辨率压缩到原来的1 / k，加快处理速度
            k = 4
            img = cv2.resize(img, (img.shape[1] // k, img.shape[0] // k))
    
            left, right, middle, visual_img, middle_line_mask = scanLR(
                img, step=5, visual=True)
    
            detect_turn_point(img, left, right, middle,
                              visual_img=visual_img, roi=0.7, step=5, visual=True)
    
            middle_lines = detect_line(middle_line_mask[img.shape[0] // 2:])
    
            middle_lines = average_lines_old(img, middle_lines)
    
            middle_line = None
    
            try:
                middle_line = middle_lines[0][0]
            except:
                continue

            middle_line = middle_lines[0][0]
            # 输出middle_lines中的middle_line斜率
            slope = c_slope(middle_line[0], middle_line[1],
                          middle_line[2], middle_line[3])
    
            img = display_line(img, middle_lines, (0, 255, 0), 2)
    
            cv2.imshow('visual_img', visual_img)
    
            # 仅放入下半部分，提高识别速度
            cv2.imshow('half_visual_img', visual_img[visual_img.shape[0] // 2:])
    
            cv2.imshow("half_middle_line_mask",
                       middle_line_mask[img.shape[0] // 2:])
    
            # ------
            cv2.imshow("main", img)
            
            # angle_to_mid_radian = -math.atan(x_offset / y_offset)  
            angle_to_mid_radian = slope
            angle_to_mid_deg = -int(angle_to_mid_radian * 180.0 / math.pi) 
            
            keyCode = cv2.waitKey(30) & 0xFF
            if keyCode == 27:  # ESC键退出
                break
        
        cap.release()
        cv2.destroyAllWindows()

    else:
        print("打开摄像头失败")

    del bot
    

    log_file.close()
