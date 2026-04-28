import sys
import os

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import threading
import time
import cv2
import numpy as np
import serial
from paddleocr import PaddleOCR # 1. 在main.py中导入PaddleOCR

# --- 任务模块导入 ---
from tasks.task1 import run_task1
from tasks.task5 import run_task5
from tasks.task7 import run_task7
from tasks.task8 import run_task8 

# --- 串口配置 ---
SERIAL_PORT_RECEIVER = '/dev/ttyUSB0'  # 用于接收任务指令
SERIAL_PORT_SENDER = '/dev/ttyACM0'    # 用于发送数据 (如任务5的坐标)
BAUDRATE = 115200

# 初始化接收串口
try:
    ser_receiver = serial.Serial(SERIAL_PORT_RECEIVER, BAUDRATE, timeout=1)
    print(f"接收串口 {SERIAL_PORT_RECEIVER} 打开成功")
except Exception as e:
    ser_receiver = None
    print(f"接收串口 {SERIAL_PORT_RECEIVER} 打开失败: {e}")

# 初始化发送串口
try:
    ser_sender = serial.Serial(SERIAL_PORT_SENDER, BAUDRATE, timeout=1)
    print(f"发送串口 {SERIAL_PORT_SENDER} 打开成功")
except Exception as e:
    ser_sender = None
    print(f"发送串口 {SERIAL_PORT_SENDER} 打开失败: {e}")


REFERENCE_RED_LIMITS = [
    (np.array([0, 100, 100]), np.array([10, 255, 255])),
    (np.array([160, 100, 100]), np.array([180, 255, 255]))
]
REFERENCE_GREEN_LIMITS = [
    (np.array([35, 100, 100]), np.array([85, 255, 255]))
]


# 任务分辨率配置，每个任务对应一个分辨率
TASK_CONFIG = {
    1: (1920, 1080),
    2: (640, 480),
    3: (640, 480),
    4: (640, 480),
    5: (1920, 1080),
    6: (1920, 1080),
    7: (1920, 1080),
    8: (1920, 1080),
}

current_task = 0
task_lock = threading.Lock()

DEBUG_MODE = 1  # 1=调试模式(手动输入)，2=串口接收

# 串口监听线程，实时接收任务编号并切换任务
def serial_listener():
    global current_task
    while True:
        if DEBUG_MODE == 1:
            user_input = input("输入任务指令 ('1'-'8', '0'): ").strip()

            if user_input in ['1', '2', '3', '5', '6', '7', '8', '0']:
                with task_lock:
                    current_task = int(user_input)
                print(f"切换到任务 {current_task}")
            else:
                print("无效的指令。")
        else:
            # 串口模式逻辑 (此处省略)
            pass

def process_task():
    global current_task
    cap = None
    last_task = -1

    while True:
        with task_lock:
            task = current_task
        
        if task == 0:
            if cap:
                cap.release()
                cap = None
                cv2.destroyAllWindows()
                last_task = 0
            time.sleep(0.1)
            continue

        if task != last_task:
            # --- 任务切换时的初始化 ---
            if task == 7:
                print(f"已切换到任务 7 - 数字识别模式")

            if cap:
                cap.release()
            cv2.destroyAllWindows()
            
            width, height = TASK_CONFIG.get(task, (640, 480))
            print(f"正在为任务 {task} 配置摄像头...")
        #------------------------------------------------------------
            cap = cv2.VideoCapture(1)
        #------------------------------------------------------------
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            if not cap or not cap.isOpened():
                print(f"错误: 无法为任务 {task} 打开摄像头。")
                time.sleep(2)
                continue
            last_task = task

        ret, frame = cap.read()
        if not ret:
            print("错误: 无法读取视频帧。")
            last_task = -1 
            continue

        continue_running = True
        if task in [1, 2, 3]:
            continue_running = run_task1(frame, ser_sender)
        elif task in [5, 6]:
            continue_running = run_task5(frame, ser_sender)
        # --- 3. 修改函数调用，传入ocr_model ---
        elif task == 7:
            continue_running = run_task7(frame, ser_sender)
        elif task == 8:
            continue_running = run_task8(frame, ser_sender)
        if not continue_running:
            print("检测到退出指令，正在关闭程序...")
            break
    
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    if ser_sender:
        ser_sender.close()
    if ser_receiver:
        ser_receiver.close()

if __name__ == "__main__":
    # 启动串口监听线程
    threading.Thread(target=serial_listener, daemon=True).start()
    # 启动视觉处理主线程
    process_task()
