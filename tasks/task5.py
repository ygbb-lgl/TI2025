import cv2
import numpy as np
from utils.measurement import calculate_distance_cm, calculate_real_size_cm
from utils.serial_utils import send_data
from utils.camera_converter import CameraConverter

# --- 相机和物理参数 ---
# 这部分参数应与task1保持一致，以确保测距和测尺寸的准确性
converter = CameraConverter()
converter.current_camera = '2_640'  
config = converter.get_camera_config()
CAMERA_MATRIX = config['mtx']
DIST_COEFFS = config['dist']

# 这个焦距F是连接真实世界和像素世界的桥梁，是测距公式的核心参数。
FOCAL_LENGTH = (CAMERA_MATRIX[0, 0] + CAMERA_MATRIX[1, 1]) / 2
KNOWN_A4_PAPER_WIDTH_MM = 210.0
KNOWN_A4_PAPER_HEIGHT_MM = 297.0

# --- 图像处理参数 ---
MIN_PAPER_AREA = 20000
MAX_PAPER_AREA = 200000
MIN_SHAPE_AREA = 500 # 最小图形像素面积，避免噪声

def run_task5(frame, ser_sender):
    if frame is None:
        return True # 返回True以继续运行

    distance_D_cm = -1.0
    size_to_send = -1.0
    
    # --- 步骤 1: 图像预处理和畸变校正 (同Task1) ---
    h, w = frame.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX, DIST_COEFFS, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFFS, None, new_camera_mtx)
    x_roi, y_roi, w_roi, h_roi = roi
    frame = undistorted_frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    binary_frame = cv2.adaptiveThreshold(blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 151, 30)
    kernel = np.ones((5, 5), np.uint8)
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel, iterations=1)

    # --- 步骤 2: 定位A4纸并计算距离D (同Task1) ---
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paper_contour = None
    paper_area = 0
    paper_pixel_width = -1.0 # 新增：用于存储A4纸像素宽度作为比例尺
    if contours:
        valid_contours = [c for c in contours if MIN_PAPER_AREA < cv2.contourArea(c) < MAX_PAPER_AREA]
        if valid_contours:
            paper_contour = max(valid_contours, key=cv2.contourArea)
            paper_area = cv2.contourArea(paper_contour)
            peri = cv2.arcLength(paper_contour, True)
            approx = cv2.approxPolyDP(paper_contour, 0.02 * peri, True)
            if len(approx) == 4:
                cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)
                points = approx.reshape(4, 2)
                sides = sorted([np.linalg.norm(points[i] - points[(i + 1) % 4]) for i in range(4)])
                paper_pixel_width = (sides[0] + sides[1]) / 2
                paper_pixel_height = (sides[2] + sides[3]) / 2
                dist_from_width = calculate_distance_cm(KNOWN_A4_PAPER_WIDTH_MM, FOCAL_LENGTH, paper_pixel_width)
                dist_from_height = calculate_distance_cm(KNOWN_A4_PAPER_HEIGHT_MM, FOCAL_LENGTH, paper_pixel_height)
                if dist_from_width > 0 and dist_from_height > 0:
                    distance_D_cm = (dist_from_width + dist_from_height) / 2.0

    # --- 步骤 3: 在A4纸内部寻找所有正方形并找到最小的一个 ---
    found_squares = []
    if paper_contour is not None and distance_D_cm > 0 and paper_pixel_width > 0:
        mask = np.zeros_like(binary_frame)
        cv2.drawContours(mask, [paper_contour], -1, 255, -1)
        masked_binary = cv2.bitwise_and(binary_frame, binary_frame, mask=mask)

        shape_contours, _ = cv2.findContours(masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in shape_contours:
            area = cv2.contourArea(cnt)
            if area < MIN_SHAPE_AREA or area > paper_area * 0.9:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

            # 判断是否为四边形
            if len(approx) == 4:
                # 绘制所有识别到的正方形轮廓 (棕色)
                cv2.drawContours(frame, [cnt], -1, (42, 42, 165), 2)

                # 计算像素边长
                sides = [np.linalg.norm(approx[i] - approx[(i + 1) % 4]) for i in range(4)]
                pixel_side_length = np.mean(sides)
                
                # --- 核心修改：使用比例尺法计算真实尺寸 ---
                # 公式: 真实尺寸 = (图形像素尺寸 / A4纸像素宽度) * A4纸真实宽度
                current_size_mm = (pixel_side_length / paper_pixel_width) * KNOWN_A4_PAPER_WIDTH_MM
                current_size_cm = current_size_mm / 10.0
                
                # 将找到的正方形信息存起来
                found_squares.append({
                    'size': current_size_cm,
                    'area': area,
                    'contour': cnt
                })

    # --- 步骤 4: 发送数据和显示结果 ---
    # 如果找到了至少一个正方形，选择面积最小的那个
    if found_squares:
        smallest_square = min(found_squares, key=lambda s: s['area'])
        size_to_send = smallest_square['size']
        
        # 高亮显示最小的正方形 (绿色)
        cv2.drawContours(frame, [smallest_square['contour']], -1, (0, 255, 0), 3)
        
        # 在最小正方形旁边显示其尺寸
        (x_b, y_b, _, _) = cv2.boundingRect(smallest_square['contour'])
        cv2.putText(frame, f"{size_to_send:.2f}cm", (x_b, y_b - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 无论是否找到，都发送当前数据
    send_data(ser_sender, distance_D_cm, size_to_send)

    # 在左上角显示距离D和最终要发送的边长x
    cv2.putText(frame, f"D: {distance_D_cm:.2f} cm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"x_min: {size_to_send:.2f} cm", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Task 5 - Find Smallest Square", frame)

    # 检查按键，如果按下 'q' 则返回 False 以退出程序
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True