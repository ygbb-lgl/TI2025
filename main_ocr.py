import cv2
from paddleocr import PaddleOCR
import time
import threading
from queue import Queue
import platform
import serial
import struct

# --- 串口配置 ---
try:
    # 注意：请根据您的设备管理器修改COM口
    ser = serial.Serial('ttyUSB0', 115200, timeout=1) 
    print("串口 COM3 打开成功")
except Exception as e:
    ser = None
    print(f"错误：无法打开串口: {e}")


# --- 0. 摄像头读取线程 ---
def camera_reader(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put_nowait(frame)
    cap.release()

# --- 1. 初始化 OCR ---
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

# --- 1.5. 分辨率配置 ---
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# --- 2. 初始化摄像头 (根据操作系统) ---
cap = None
if platform.system() == 'Linux':
    print(f"系统: Linux. 正在配置GStreamer管道...")
    gst_str = (
        f"v4l2src device=/dev/video0 ! "
        f"video/x-raw,width={FRAME_WIDTH},height={FRAME_HEIGHT},framerate=30/1 ! "
        f"videoconvert ! "
        f"appsink"
    )
    print(f"GStreamer Pipeline: {gst_str}")
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
else:
    print(f"系统: Windows/Other. 正在配置摄像头...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print(f"摄像头分辨率设置为: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

if not cap or not cap.isOpened():
    print("错误：无法打开摄像头。")
    exit()

# 创建队列和摄像头读取线程
frame_queue = Queue(maxsize=1)
reader_thread = threading.Thread(target=camera_reader, args=(cap, frame_queue))
reader_thread.daemon = True
reader_thread.start()

print("摄像头和处理线程已启动...")

# 用于计算FPS
prev_frame_time = 0

# --- 3. 实时识别循环 (主线程) ---
while True:
    if frame_queue.empty():
        time.sleep(0.01)
        continue

    # 从队列中获取最新的帧
    frame = frame_queue.get()

    # 将图像帧上下颠倒
    flipped_frame = cv2.flip(frame, 0)
    
    # === 关键修改：颜色反转 ===
    inverted_frame = 255 - flipped_frame

    # --- OCR 识别 ---
    # 使用翻转+颜色反转后的图像进行识别
    result = ocr.ocr(inverted_frame)  # 使用反转后的帧

    # --- 处理并绘制结果 ---
    if result and result[0]:
        print("OCR原始结果：", result[0])
        for line in result[0]:
            print("line内容：", line)
            # box = line[0]
            # 检查 box 是否为长度为4的点列表
            # if not (isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in box)):
            #     print(f"警告：box格式异常，跳过本行，内容为: {box}")
            #     continue

            # if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
            #     text = line[1][0]
            #     confidence = line[1][1]
            # else:
            #     text = str(line[1])
            #     confidence = 1.0
            
            # # 获取整行文本框的坐标
            # top_left = box[0]
            # top_right = box[1]
            # bottom_right = box[2]
            
            # # 计算文本框的总宽度
            # line_width = top_right[0] - top_left[0]

            # if len(text) > 0:
            #     avg_char_width = line_width / len(text)

            #     for i, char in enumerate(text):
            #         # 5. 只发送数字
            #         if not char.isdigit():
            #             continue

            #         # 估算当前字符的左上角和右下角坐标
            #         char_x_min = int(top_left[0] + i * avg_char_width)
            #         char_y_min = int(top_left[1])
            #         char_x_max = int(top_left[0] + (i + 1) * avg_char_width)
            #         char_y_max = int(bottom_right[1])

            #         # 坐标重映射到原始帧用于绘制
            #         draw_y_min = FRAME_HEIGHT - 1 - char_y_max
            #         draw_y_max = FRAME_HEIGHT - 1 - char_y_min

            #         # 在原始图像上绘制矩形框
            #         cv2.rectangle(frame, (char_x_min, draw_y_min), (char_x_max, draw_y_max), (0, 0, 255), 2)
                    
            #         # 在原始图像上标注字符
            #         cv2.putText(frame, char, (char_x_min, draw_y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            #         # --- 通过串口发送数据 ---
            #         if ser:
            #             try:
            #                 center_u = char_x_min + (char_x_max - char_x_min) // 2
            #                 frame_header1 = 0xAA
            #                 frame_header2 = 0x55
            #                 data_packet = struct.pack('<BBcH', frame_header1, frame_header2, char.encode('utf-8'), center_u)
            #                 ser.write(data_packet)
            #                 print(f"发送数据: char='{char}', u={center_u}")
            #             except Exception as e:
            #                 print(f"串口发送失败: {e}")


    # --- 计算并显示FPS ---
    new_frame_time = time.time()
    if (new_frame_time - prev_frame_time) > 0:
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- 显示结果帧 ---
    # 先只显示原始帧
    cv2.imshow("Real-time OCR - Press 'q' to quit", frame)

    # 检测按键，如果按下 'q' 则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. 释放资源 ---
cv2.destroyAllWindows()
if ser:
    ser.close()
    print("串口已关闭")