import cv2
import time

# 调用摄像头
cap = cv2.VideoCapture(0)

if cap.isOpened():
    window_handle = cv2.namedWindow("D435", cv2.WINDOW_AUTOSIZE)

    # 逐帧显示
    while cv2.getWindowProperty("D435", 0) >= 0:
        ret_val, img = cap.read()
        cv2.imshow("D435", img)
        print(img.shape)
        keyCode = cv2.waitKey(30) & 0xFF
        if keyCode == 27:  # ESC键退出
            break
        # 按s捕捉一张图片, 用系统时间命名
        if keyCode == ord('s'):
            cv2.imwrite("./img/" + time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime()) + ".jpg", img)
            print("save a picture")

            

cap.release()
cv2.destroyAllWindows()
