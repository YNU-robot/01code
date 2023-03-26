import cv2
import time
import time
from Transbot_Lib import Transbot

# 调用摄像头
cap = cv2.VideoCapture(0)

# bot对象
bot = Transbot()
# 启动接收数据
bot.create_receive_threading()

# 开启自动发送数据
# enable=True，底层扩展板会每隔40毫秒发送一次数据。enable=False，则不发送。
# forever=True永久保存，=False临时作用。
# Enable automatic data sending  
# enable=True, the underlying expansion board will send data every 40 milliseconds.  If enable=False, the port is not sent.  
# forever=True for permanent, =False for temporary.  
enable = True
bot.set_auto_report_state(enable, forever=False)
deta = 0.1 
id = 0

if cap.isOpened():
    window_handle = cv2.namedWindow("D435", cv2.WINDOW_AUTOSIZE)


    # 逐帧显示
    while cv2.getWindowProperty("D435", 0) >= 0:
        ret_val, img = cap.read()
        # cv2.imshow("D435", img)

        try:
            v, a = bot.get_motion_data()
            print("speed:", v, a)
            bot.clear_auto_report_data()
            time.sleep(deta)
            print("speed: "+ str(v) + ' ' + str(a))
            cv2.imwrite("./traindataa/" + f"{id}_{a}" + ".jpg", img)
            cv2.imwrite("./traindatava/" + f"{id}_{v}_{a}" + ".jpg", img)
            print(f"save picture{id}")
        except Exception:
            print("Exception")
            break
        keyCode = cv2.waitKey(30) & 0xFF
        if keyCode == 27:  # ESC键退出
            break
        # 按s捕捉一张图片, 用系统时间命名
        if keyCode == ord('s'):
            cv2.imwrite("./img/" + time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime()) + ".jpg", img)
            print("save a picture")
        id+=1

            
# 关闭自动发送数据
# enable=True，底层扩展板会每隔40毫秒发送一次数据。enable=False，则不发送。
# forever=True永久保存，=False临时作用。
# Disable automatic sending of data  
# enable=True, the underlying expansion board will send data every 40 milliseconds.  If enable=False, the port is not sent.  
# forever=True for permanent, =False for temporary.  
enable = False
bot.set_auto_report_state(enable, forever=False)

# 清除单片机自动发送过来的缓存数据
# Clear the cache data automatically sent by the MCU
bot.clear_auto_report_data()

del bot
cap.release()
cv2.destroyAllWindows()
