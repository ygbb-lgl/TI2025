import cv2
from paddleocr import PaddleOCR
import numpy as np

# --- 1. 初始化 PaddleOCR ---
# 这个过程可能需要一些时间来下载和加载模型
print("正在加载 PaddleOCR 模型, 请稍候...")
try:
    # lang='ch' 表示使用中文和英文的识别模型
    # use_gpu=False 表示使用CPU，如果您的环境配置好了CUDA，可以设置为True以获得加速
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False, show_log=False)
    print("PaddleOCR 模型加载成功。")
except Exception as e:
    print(f"错误：加载 PaddleOCR 模型失败: {e}")
    exit()

# --- 2. 初始化摄像头 ---
# 0 代表默认的摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("错误：无法打开摄像头。")
    exit()

print("摄像头已启动，按 'q' 键退出。")

# --- 3. 主循环：实时读取、识别和显示 ---
while True:
    # 读取一帧视频
    ret, frame = cap.read()
    if not ret:
        print("错误：无法读取视频帧。")
        break

    # --- 4. 执行OCR识别 ---
    # ocr.ocr() 函数会返回一个列表，每个元素包含识别到的文字、坐标和置信度
    result = ocr.ocr(frame, cls=True)

    # 创建一个副本用于绘制结果，避免在原始帧上修改
    result_frame = frame.copy()

    # --- 5. 解析并绘制识别结果 ---
    if result and result[0]:
        # 遍历所有识别到的文本行
        for line in result[0]:
            # line 结构: [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ('文本', 置信度)]
            box = line[0]
            text = line[1][0]
            confidence = line[1][1]

            # 将坐标点转换为整数，并构造成Numpy数组以用于绘制
            points = np.array(box).astype(int)

            # 如果置信度高于一个阈值（例如0.5），才显示结果
            if confidence > 0.5:
                # 绘制文本的边界框 (绿色)
                cv2.polylines(result_frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # 在边界框左上角上方显示识别出的文本和置信度 (红色)
                # 注意：cv2.putText不支持直接绘制中文，在Windows上可能显示为乱码。
                # 在Linux上可能需要配置字体才能正常显示。这是一个OpenCV的限制。
                cv2.putText(result_frame, f"{text}", 
                            (points[0][0], points[0][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 0, 255), 2)

    # --- 6. 显示处理后的视频帧 ---
    cv2.imshow("实时OCR检测 (按 'q' 退出)", result_frame)

    # --- 7. 检测退出键 ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 8. 释放资源 ---
cap.release()
cv2.destroyAllWindows()
print("程序已退出。")