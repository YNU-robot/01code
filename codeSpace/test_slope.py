
from cv2 import sqrt
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
        print("垂直线")
        # 垂直线
        theta_rad = math.pi / 2
        # res = theta * (180 / math.pi)
    return theta_rad


print(c_slope(0, 0, 1, 1))
print(math.tan(math.pi / 4))
print(c_slope(0, 0, 1, math.sqrt(3) / 3 ))
print(c_slope(0, 0, 0, 1)) 