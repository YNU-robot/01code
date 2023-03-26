from Transbot_Lib import Transbot
import time

# 创建Transbot对象 bot
# Create Transbot object as bot
bot = Transbot()


# 普通摄像头的二自由度云台控制。
# 注意：深度相机不要运行此命令，由于没有限制角度可能会撞上其他东西。
# General camera two degrees of freedom control head.
# Note: Do not run this command for Astra camera, as there is no limit to the Angle you may bump into something else.
# 90为原位， 值范围在30-150之间
def pwm_servo(x, y):
    bot.set_pwm_servo(1, x)
    bot.set_pwm_servo(2, y)
    return x, y


# 复位
pwm_servo(120, 110)
time.sleep(1)
pwm_servo(87, 110)

bot.set_car_motion(0, 0)
# 控制探照灯熄灭
# Control searchlight to go out
light = 0
bot.set_floodlight(light)
time.sleep(1)
del bot
