import sys
import os

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import threading
import time
import cv2
import numpy as np
import serial
from paddleocr import PaddleOCR

# --- 修改：导入您提供的INA219驱动库 ---
try:
    import DFRobot_INA219
    INA219_AVAILABLE = True
except ImportError:
    INA219_AVAILABLE = False
    print("警告: 'DFRobot_INA219.py' 库未找到或其依赖(smbus)缺失。")
    print("请确保 DFRobot_INA219.py 与 main.py 在同一目录，并已安装smbus。")

# --- 任务模块导入 ---
from tasks.task1 import run_task1
from tasks.task5 import run_task5
from tasks.task7 import run_task7, set_target_number as set_target_number_7
from tasks.task8 import run_task8
from utils.serial_utils import send_data

# --- 全局OCR模型初始化 ---
print("正在主程序中初始化 PaddleOCR 模型...")
try:
    ocr_model = PaddleOCR(
        use_angle_cls=True,      # 处理可能倾斜的数字
        lang='ch',               # 英文模式，更适合数字识别
        use_gpu=True
    )
    print("--- PaddleOCR 模型初始化成功 ---")
    OCR_AVAILABLE = True
except Exception as e:
    ocr_model = None
    OCR_AVAILABLE = False
    print(f"无法初始化PaddleOCR，程序可能无法正常执行任务7: {e} ---")

# --- 用于存储传感器数据的全局变量和锁 ---
current_A = 0.0
power_W = 0.0
max_power_W = 0.0
mcu_data_lock = threading.Lock()

# --- 串口和I2C配置 ---
#SERIAL_PORT_HMI = 'COM7'  # 在Windows上调试时使用
SERIAL_PORT_HMI = '/dev/ttyUSB0' # 在Jetson上使用
BAUDRATE = 115200

# --- 新增：INA219传感器配置 ---
I2C_BUS_NUMBER = 1  # Jetson Nano 通常使用 I2C bus 1
INA219_I2C_ADDRESS = 0x45 # 传感器的I2C地址

# --- HMI串口初始化 ---
try:
    ser_hmi = serial.Serial(SERIAL_PORT_HMI, BAUDRATE, timeout=10)
    print(f"HMI串口 {SERIAL_PORT_HMI} 打开成功")
except Exception as e:
    ser_hmi = None
    print(f"HMI串口 {SERIAL_PORT_HMI} 打开失败: {e}")

# --- 移除：不再需要MCU串口的初始化 ---

REFERENCE_RED_LIMITS = [
    (np.array([0, 100, 100]), np.array([10, 255, 255])),
    (np.array([160, 100, 100]), np.array([180, 255, 255]))
]
REFERENCE_GREEN_LIMITS = [
    (np.array([35, 100, 100]), np.array([85, 255, 255]))
]

TASK_CONFIG = {
    1: (1920, 1080), 2: (1920, 1080), 3: (1920, 1080), 4: (1920, 1080),
    5: (1920, 1080), 6: (1920, 1080), 7: (1920, 1080), 8: (1920, 1080),
}

current_task = 0
task_lock = threading.Lock()
target_numbers = {7: -1}
DEBUG_MODE = 0

# --- 替换：新的INA219传感器监听线程 ---
def ina219_listener():
    """在一个独立的线程中运行，每5秒通过I2C读取INA219传感器数据。"""
    global current_A, power_W, max_power_W

    if not INA219_AVAILABLE:
        print("INA219库不可用，无法启动传感器监听线程。")
        return # 如果库不存在，则直接退出线程

    # 初始化INA219传感器对象
    ina = DFRobot_INA219.INA219(bus=I2C_BUS_NUMBER, addr=INA219_I2C_ADDRESS)

    # 循环尝试初始化传感器，直到成功
    while True:
        if ina.begin():
            print("INA219传感器初始化成功。")
            break
        else:
            print("INA219传感器初始化失败，1秒后重试...")
            time.sleep(1)
    
    # 可选：如果需要，可以在这里进行线性校准
    # ina.linear_cal(1000, 1000) # 示例：输入1000mA，实际也是1000mA

    while True:
        try:
            # 从传感器读取数据 (单位是 mA 和 mW)
            current_mA = ina.get_current_mA()
            power_mW = ina.get_power_mW()

            # --- 新增调试信息 ---
            print(f"[DEBUG] INA219 原始读数: Current={current_mA} mA, Power={power_mW} mW")

            # 将单位从 mA/mW 转换为 A/W
            current_val = current_mA / 1000.0
            power_val = power_mW / 1000.0

            # 使用锁安全地更新全局变量
            with mcu_data_lock:
                current_A = current_val
                power_W = power_val
                # 在Python端计算最大功率
                if power_W > max_power_W:
                    max_power_W = power_W
            
            # (可选) 打印调试信息
            # print(f"INA219读数: I={current_A:.3f}A, P={power_W:.3f}W, P_max={max_power_W:.3f}W")

        except Exception as e:
            # 捕获所有可能的I2C通信错误
            print(f"读取INA219传感器时出错: {e}")
            with mcu_data_lock: # 通信失败时，将数据清零
                current_A = 0.0
                power_W = 0.0
        
        # 等待5秒再进行下一次读取
        time.sleep(0.5)

# HMI串口监听线程 (保持不变)
def serial_listener():
    global current_task
    while True:
        if DEBUG_MODE == 1:
            user_input = input("输入任务指令 ('1'-'8', '0'退出, '7'带数字): ").strip()
            
            if user_input in ['7']:
                task_num = int(user_input)
                while True:
                    try:
                        number_input = input(f"请输入任务 {task_num} 的目标数字: ").strip()
                        target_num = int(number_input)
                        # 更新全局字典和特定任务的设置
                        target_numbers[task_num] = target_num
                        if task_num == 7:
                            set_target_number_7(target_num)
                        print(f"任务 {task_num} 目标编号已设置为: {target_num}")
                        break
                    except ValueError:
                        print("无效的数字，请重新输入。")
                with task_lock:
                    current_task = task_num
                print(f"已切换到任务 {task_num}。")

            elif user_input in ['1', '2', '3', '4', '5', '6', '8', '0']:
                with task_lock:
                    current_task = int(user_input)
                print(f"切换到任务 {current_task}")
            else:
                print("无效的指令。")
            pass
        else:
            if ser_hmi and ser_hmi.in_waiting > 0:
                byte_data = ser_hmi.read(1)
                # 打印数据格式
                # print(f"接收到串口数据: {byte_data.hex()}")

                if byte_data:
                    task_num = int.from_bytes(byte_data, byteorder='big', signed=False)

                    if 1 <= task_num <= 9:
                        if task_num == 7:
                            print("已接收到任务7指令，正在等待目标数字...")
                            # 阻塞式读取，等待任务7的目标数字
                            target_byte_data = ser_hmi.read(1)
                            if target_byte_data:
                                target_num = int.from_bytes(target_byte_data, byteorder='big', signed=False)
                                # 更新全局字典和特定任务的设置
                                target_numbers[7] = target_num
                                set_target_number_7(target_num)
                                print(f"任务7目标数字已设置为: {target_num}")
                                with task_lock:
                                    current_task = 7
                                print(f"切换到任务 {current_task}")
                            else:
                                # 如果没有读到数据（例如超时），可以设置一个默认行为
                                print("警告：读取任务7目标数字超时或失败。")
                        else:
                            with task_lock:
                                current_task = task_num
                            print(f"切换到任务 {current_task}")
            else:
                time.sleep(0.01)

def process_task():
    global current_task
    
    print("正在初始化摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("致命错误: 无法打开摄像头。程序将退出。")
        return

    initial_width, initial_height = TASK_CONFIG.get(1, (1920, 1080))
    print(f"正在配置摄像头分辨率为: {initial_width}x{initial_height}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, initial_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, initial_height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    last_task = -1

    while True:
        with task_lock:
            task = current_task
        
        with mcu_data_lock:
            current_display = current_A
            power_display = power_W
            max_power_display = max_power_W

        # 假设page0的t5显示电流, t7显示功率, t9显示最大功率
        send_data(ser_hmi, "t5", "txt", f"{current_display:.2f} A")
        send_data(ser_hmi, "t7", "txt", f"{power_display:.2f} W")
        send_data(ser_hmi, "t9", "txt", f"{max_power_display:.2f} W")

        if task == 0:
            if last_task != 0:
                cv2.destroyAllWindows()
                print("任务暂停，摄像头保持开启状态。")
                last_task = 0
            time.sleep(0.1)
            continue

        if task != last_task:
            if task == 7:
                set_target_number_7(target_numbers[7])
            cv2.destroyAllWindows()
            print(f"已切换到任务 {task}。")
            last_task = task

        ret, frame = cap.read()
        if not ret:
            print("错误: 无法读取视频帧。")
            time.sleep(0.5)
            continue

        continue_running = True
        if task in [1, 2, 3]:
            continue_running = run_task1(frame, ser_hmi)
        elif task in [5, 6]:
            continue_running = run_task5(frame, ser_hmi)
        elif task == 7:
            continue_running = run_task7(frame, ser_hmi, None, ocr_model, DEBUG_MODE)
        elif task == 8:
            continue_running = run_task8(frame, ser_hmi)
        
        # --- 更新发送到HMI的数据 ---
        

        
        if not continue_running:
            print("检测到退出指令，正在关闭程序...")
            break
    
    print("正在释放摄像头资源...")
    cap.release()
    cv2.destroyAllWindows()
    if ser_hmi:
        ser_hmi.close()

if __name__ == "__main__":
    # 启动HMI串口监听线程
    threading.Thread(target=serial_listener, daemon=True).start()
    
    # --- 修改：启动新的INA219传感器监听线程 ---
    threading.Thread(target=ina219_listener, daemon=True).start()

    # 启动视觉处理主线程
    process_task()
