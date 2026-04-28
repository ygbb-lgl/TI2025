import cv2
import numpy as np
from utils.camera_converter import CameraConverter 
from utils.measurement import calculate_distance_cm, calculate_real_size_cm
from utils.serial_utils import send_data


# 1. 相机参数
converter = CameraConverter()
converter.current_camera = '2_640'  
config = converter.get_camera_config()

# 将获取到的相机内参矩阵
CAMERA_MATRIX = config['mtx']
# 将获取到的相机畸变系数
DIST_COEFFS = config['dist']

# 这个焦距F是连接真实世界和像素世界的桥梁，是测距公式的核心参数。
FOCAL_LENGTH = (CAMERA_MATRIX[0, 0] + CAMERA_MATRIX[1, 1]) / 2

# --- 2. 参考物真实尺寸 (单位: 毫米) ---
# A4纸的真实宽度，单位毫米。
KNOWN_A4_PAPER_WIDTH_MM = 210.0
# A4纸的真实高度，单位毫米。
KNOWN_A4_PAPER_HEIGHT_MM = 297.0


ADAPTIVE_THRESH_BLOCK_SIZE = 151 
ADAPTIVE_THRESH_C = 30       

MIN_PAPER_AREA = 2000  # A4纸在最远处时的最小面积
MAX_PAPER_AREA = 200000 # A4纸在最近处时的最大面积


def run_task1(frame, ser_sender):
    if frame is None:
        print("任务1: 接收到空帧")
        return

    # --- 步骤 1: 图像预处理 ---
    
    h, w = frame.shape[:2]
    # 计算用于畸变校正的最佳相机矩阵和感兴趣区域(ROI)
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX, DIST_COEFFS, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFFS, None, new_camera_mtx)
    x_roi, y_roi, w_roi, h_roi = roi
    frame = undistorted_frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # THRESH_BINARY_INV：反转二值化，使A4纸上的黑色边框和图形变成白色(255)，白色背景变成黑色(0)。
    binary_frame = cv2.adaptiveThreshold(
        blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C
    )

    kernel = np.ones((5, 5), np.uint8)
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imshow('binary',binary_frame)

    # --- 步骤 2: 测量距离 D  ---
    distance_D_cm = -1.0 
    paper_area = 0 
    paper_contour = None # <-- 新增：在此处初始化变量
    paper_pixel_width = -1.0 # 新增：用于存储A4纸像素宽度作为比例尺

    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if MIN_PAPER_AREA < area < MAX_PAPER_AREA:
                valid_contours.append(contour)

        if valid_contours:
            paper_contour = max(valid_contours, key=cv2.contourArea)
            paper_area = cv2.contourArea(paper_contour)

            peri = cv2.arcLength(paper_contour, True)
            approx = cv2.approxPolyDP(paper_contour, 0.02 * peri, True) # 使用一个较小的逼近值

            if len(approx) == 4:
                # 1.1: 识别A4纸的外轮廓用蓝色线画出
                cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2) 
                points = approx.reshape(4, 2)
                
                # 1.2: 它的四个角点用绿色画出
                for point in points:
                    cv2.circle(frame, tuple(point), 5, (0, 255, 0), -1)
                
                sides = sorted([np.linalg.norm(points[i] - points[(i + 1) % 4]) for i in range(4)])
                
                paper_pixel_width = (sides[0] + sides[1]) / 2
                paper_pixel_height = (sides[2] + sides[3]) / 2

                dist_from_width = calculate_distance_cm(KNOWN_A4_PAPER_WIDTH_MM, FOCAL_LENGTH, paper_pixel_width)
                dist_from_height = calculate_distance_cm(KNOWN_A4_PAPER_HEIGHT_MM, FOCAL_LENGTH, paper_pixel_height)
                
                if dist_from_width > 0 and dist_from_height > 0:
                    distance_D_cm = (dist_from_width + dist_from_height) / 2.0

    # --- 步骤 3: 测量图形尺寸 x (已修改为识别实心图形) ---
    size_to_send = -1.0
    found_shapes = []

    # 只在成功识别A4纸并计算出距离后，才进行图形测量
    if paper_contour is not None and distance_D_cm > 0 and paper_pixel_width > 0:
        # 创建一个掩码，只在A4纸内部寻找轮廓，避免找到A4纸外的干扰
        mask = np.zeros_like(binary_frame)
        cv2.drawContours(mask, [paper_contour], -1, 255, -1)
        
        # 将掩码应用于二值图，确保我们只处理A4纸内部的像素
        inner_binary = cv2.bitwise_and(binary_frame, binary_frame, mask=mask)
        cv2.imshow('inner_binary', inner_binary)

        # 从A4纸内部寻找所有独立的外部轮廓（即实心图形）
        shape_contours, _ = cv2.findContours(inner_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in shape_contours:
            # 过滤掉面积过小（噪声）或过大（可能是A4纸边框本身）的轮廓
            area = cv2.contourArea(contour)
            if area < 500 or area > paper_area * 0.9:
                continue

            pixel_length_for_calc = 0

            # 对轮廓进行多边形逼近以判断形状
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

            # 根据顶点数量判断形状并计算尺寸
            if len(approx) == 3: # 三角形
                sides = [np.linalg.norm(approx[i] - approx[(i + 1) % 3]) for i in range(3)]
                pixel_length_for_calc = np.mean(sides)
            
            elif len(approx) == 4: # 正方形
                sides = [np.linalg.norm(approx[i] - approx[(i + 1) % 4]) for i in range(4)]
                pixel_length_for_calc = np.mean(sides)

            elif len(approx) > 6: # 认为是圆形
                (x, y), radius = cv2.minEnclosingCircle(contour)
                pixel_length_for_calc = radius * 2 # 直径

            # 如果成功识别出形状并计算了像素尺寸
            if pixel_length_for_calc > 0:
                # --- 核心修改：使用比例尺法计算真实尺寸 ---
                # 公式: 真实尺寸 = (图形像素尺寸 / A4纸像素宽度) * A4纸真实宽度
                current_size_mm = (pixel_length_for_calc / paper_pixel_width) * KNOWN_A4_PAPER_WIDTH_MM
                current_size_cm = current_size_mm / 10.0
                
                found_shapes.append({
                    'size': current_size_cm,
                    'area': area,
                    'contour': contour
                })

                # 绘制轮廓 (棕色)
                cv2.drawContours(frame, [contour], -1, (42, 42, 165), 2)

                # 绘制角点 (紫色)
                for point in approx:
                    cv2.circle(frame, tuple(point[0]), 4, (128, 0, 128), -1)
                
                # 在图形旁边显示测量的尺寸
                (x_b, y_b, _, _) = cv2.boundingRect(contour)
                cv2.putText(frame, f"{current_size_cm:.2f}cm", (x_b, y_b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # --- 步骤 4: 发送数据和显示结果 ---
    # 如果找到了至少一个图形，选择面积最大的那个来发送数据
    if found_shapes:
        largest_shape = max(found_shapes, key=lambda s: s['area'])
        size_to_send = largest_shape['size']

    # 调用函数，将距离和最大图形的尺寸发送出去
    send_data(ser_sender, distance_D_cm, size_to_send)
 
    # 在左上角显示距离D和最终要发送的边长/直径x
    cv2.putText(frame, f"D: {distance_D_cm:.2f} cm", (20, 40), # 修改：在左上角显示距离
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"x: {size_to_send:.2f} cm", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Task 1", frame)
    
    # 修改：检查按键，如果按下 'q' 则返回 False
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True