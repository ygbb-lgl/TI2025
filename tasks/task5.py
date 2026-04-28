import cv2
import numpy as np
from utils.measurement import calculate_distance_cm, calculate_real_size_cm
from utils.serial_utils import send_data
from utils.camera_converter import CameraConverter

# --- 相机和物理参数 ---
# 这部分参数应与task1保持一致，以确保测距和测尺寸的准确性
converter = CameraConverter()
converter.current_camera = '2_1080'  # 使用1080p摄像头
config = converter.get_camera_config()
CAMERA_MATRIX = config['mtx']
DIST_COEFFS = config['dist']

KNOWN_A4_PAPER_WIDTH_MM = 210.0
KNOWN_A4_PAPER_HEIGHT_MM = 297.0

# --- 图像处理参数 ---
MIN_PAPER_AREA = 3000
MAX_PAPER_AREA = 30000 # PnP适用范围更广，适当调大
MIN_SHAPE_AREA = 500 # 最小图形像素面积，避免噪声

def order_points(pts):
    """
    对找到的四个角点进行排序，确保顺序为：左上、右上、右下、左下。
    这是 PnP 算法正确工作的关键步骤。
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # 左上角的点 x+y 的和最小
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    
    # 右下角的点 x+y 的和最大
    rect[2] = pts[np.argmax(s)]
    
    # 右上角的点 y-x 的差最小
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    
    # 左下角的点 y-x 的差最大
    rect[3] = pts[np.argmax(diff)]
    
    return rect

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
    #提取图像中心区域作为ROI (按要求保留)
    h, w = frame.shape[:2]
    roi_x_start = 860  
    roi_x_end   = w - roi_x_start   
    roi_y_start = 350
    roi_y_end   = h - roi_y_start
    # 确保裁剪区域不超出图像边界
    roi_x_start, roi_y_start = max(0, roi_x_start), max(0, roi_y_start)
    roi_x_end, roi_y_end = min(w, roi_x_end), min(h, roi_y_end)
    if roi_x_start >= roi_x_end or roi_y_start >= roi_y_end:
        print("无效的ROI区域，无法裁剪图像。")
        # 如果ROI无效，则不进行裁剪，直接使用去畸变后的图像
        pass
    else:
        frame = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    # roi_x_start = 600
    # roi_x_end   = w - roi_x_start
    # roi_y_start = 350
    # roi_y_end   = h - roi_y_start
    # frame = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    # #对ROI裁剪后的图像进行放大（如3倍）
    # h_roi, w_roi = frame.shape[:2]
    # scale_factor = 3
    # frame = cv2.resize(frame, (int(w_roi * scale_factor), int(h_roi * scale_factor)), interpolation=cv2.INTER_LINEAR)


    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # --- 步骤1.1: 外部A4纸定位用自适应阈值 ---
    binary_frame = cv2.adaptiveThreshold(
    blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV, 21, 8
    )
    # --- 步骤 1.5: 形态学操作 ---
    kernel = np.ones((3, 1), np.uint8)
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- 步骤 2: 定位A4纸并计算距离D ---
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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
                # --- 核心修改：使用 PnP 算法替换焦距法 ---
                
                # 1. 定义世界坐标系中的3D点 (A4纸的四个角点)
                object_points = np.array([
                    [0, 0, 0],
                    [KNOWN_A4_PAPER_WIDTH_MM, 0, 0],
                    [KNOWN_A4_PAPER_WIDTH_MM, KNOWN_A4_PAPER_HEIGHT_MM, 0],
                    [0, KNOWN_A4_PAPER_HEIGHT_MM, 0]
                ], dtype="float32")

                # 2. 获取图像坐标系中的2D点，并排序
                image_points = approx.reshape(4, 2).astype('float32')
                sorted_image_points = order_points(image_points)

                # 3. 调用 solvePnP
                success, rvec, tvec = cv2.solvePnP(object_points, sorted_image_points, CAMERA_MATRIX, DIST_COEFFS)

                if success:
                    # tvec[2] 就是我们需要的深度信息（距离），单位是毫米
                    distance_D_cm = tvec[2][0] / 10.0
                    distance_D_cm = distance_D_cm - 5.0  # <--- 修改这里，进行线性校正
                    
                    # 可选：在图像上绘制坐标轴以可视化姿态
                    axis_points = np.float32([[0,0,0], [50,0,0], [0,50,0], [0,0,-50]]).reshape(-1,3)
                    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
                    origin = tuple(map(int, imgpts[0].ravel()))
                    cv2.line(frame, origin, tuple(map(int, imgpts[1].ravel())), (255,0,0), 3) # X轴
                    cv2.line(frame, origin, tuple(map(int, imgpts[2].ravel())), (0,255,0), 3) # Y轴
                    cv2.line(frame, origin, tuple(map(int, imgpts[3].ravel())), (0,0,255), 3) # Z轴

                # --- 保留像素宽度计算，用于后续测量内部图形尺寸 ---
                sides = sorted([np.linalg.norm(sorted_image_points[i] - sorted_image_points[(i + 1) % 4]) for i in range(4)])
                paper_pixel_width = (sides[0] + sides[1]) / 2
                
                cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)

    # --- 步骤 3: 在A4纸内部寻找所有正方形并找到最小的一个 ---
    found_squares = []
    if paper_contour is not None and distance_D_cm > 0 and paper_pixel_width > 0:
        # 1. 获取A4纸轮廓的最小正矩形边界框
        x, y, w, h = cv2.boundingRect(paper_contour)

        # 2. 根据边界框裁剪二值图，得到A4纸区域
        #    这可以确保我们只处理A4纸内部的内容
        inner_gray = gray_frame[y:y+h, x:x+w]
        inner_blur = cv2.GaussianBlur(inner_gray, (5, 5), 0)
        _, inner_binary = cv2.threshold(
            inner_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # --- 在新窗口中显示这个被正确处理的内部区域 ---
        display_inner_binary = cv2.resize(inner_binary, (0,0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('inner_binary', display_inner_binary)
        

        shape_contours, _ = cv2.findContours(inner_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        #找到图像后选取角度在88-92度之间的角度所对应的顶点按顺时针依次标号排序，使用数组记录张角朝向面向coutour内部的顶点标号，
        #最后得到一个包含正方形顶点的数组，获取数组的大小，遍历数组，标号相差1的被分为一类，视为同一个正方形的顶点（记为分类方法1）。最后剩下的顶点都是没有邻近点的孤立点
        #对于孤立点，假如两个孤立点的和为数组的大小，就认为这两个孤立点是同一个正方形的顶点（记为分类方法2）。
        #采用方法1分类的正方形，将标号相差1的顶点像素坐标作差，求出点距离，用比例尺法计算真实尺寸，结果存入新数组
        #采用方法2分类的正方形，求出孤立点坐标像素距离，乘以二分之根号二得到距离，用比例尺法计算真实尺寸，结果存入新数组
        #最后返回新数组中最小的数作为最小正方形的边长
        #每个轮廓计算一遍，比较大小
        all_squares = []#整个图像的正方形
        for cnt in shape_contours:
            area = cv2.contourArea(cnt)
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)  # 用绿色线条绘制当前轮廓边界
            if area < MIN_SHAPE_AREA or area > paper_area * 0.7:
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)#找到拐点
            for pt in approx.reshape(-1, 2):
                pt_on_frame = (int(pt[0] + x), int(pt[1] + y))
                cv2.circle(frame, pt_on_frame, 5, (255, 0, 0), -1)  # 蓝色圆点

            print(f"轮廓顶点数: {len(approx)}")
            #凸包顺时针
            hull = cv2.convexHull(approx, returnPoints=True)
            pts = hull.reshape(-1, 2)

            if cv2.contourArea(pts) > 0:
                pts = pts[::-1]  # 逆时针则反转为顺时针
            
            #标注凸包顶点
            for pt in pts:
                pt_on_frame = (int(pt[0] + x), int(pt[1] + y))
                cv2.rectangle(frame, (pt_on_frame[0]-3, pt_on_frame[1]-3), (pt_on_frame[0]+3, pt_on_frame[1]+3), (0, 255, 0), -1)  # 绿色方块
            m = len(approx)
            n = len(pts)
            print(f"轮廓顶点数: {n}")
            valid_angle_indices = [None] * len(approx)  #存顶点
            count = 0
            for i in range(m):
                p_prev = approx[i - 1].squeeze()
                p_curr = approx[i].squeeze()
                p_next = approx[(i + 1) % m].squeeze()
                v1 = p_prev - p_curr
                v2 = p_next - p_curr
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                #if 50 <= angle <= 130:
                cross_z = v1[0]*v2[1] - v1[1]*v2[0]
                if cross_z > 0:  # 确保是向内的角
                    print(p_curr)
                    valid_angle_indices[i] = p_curr     #存储角度有效顶点
                    count+=1  #有效顶点数
            # 新数组filtered_vertices，顺序与pts一致，不满足条件的为None
            filtered_vertices = valid_angle_indices  # 顺序和长度都与pts一致
            print(filtered_vertices)
            print(f"有效顶点数: {count}")
            #标注所有有效顶点
            for pt in filtered_vertices:
                if pt is not None:
                    pt_on_frame = (int(pt[0] + x), int(pt[1] + y))
                    cv2.circle(frame, pt_on_frame, 6, (0, 0, 255), -1)  # 红色圆点

            #存储正方形边长
            side_lengths = []
            for i in range(m):
                pt1 = filtered_vertices[i]
                pt2 = filtered_vertices[(i + 1) % m]  # 循环相邻
                if pt1 is not None and pt2 is not None:
                    dist = np.linalg.norm(pt1 - pt2)
                    side_lengths.append(dist)
            #孤立点处理        
            isolated_indices = []
            for i in range(m):
                pt = filtered_vertices[i]
                prev_pt = filtered_vertices[i - 1]
                next_pt = filtered_vertices[(i + 1) % m]
                if pt is not None and prev_pt is None and next_pt is None:
                    isolated_indices.append(i)

            # 两两组合，判断下标和是否等于m
            for i in range(len(isolated_indices)):
                for j in range(i + 1, len(isolated_indices)):
                    idx1 = isolated_indices[i]
                    idx2 = isolated_indices[j]
                    if idx1 + idx2 == m:
                        pt1 = filtered_vertices[idx1]
                        pt2 = filtered_vertices[idx2]
                        if pt1 is not None and pt2 is not None:
                            dist = np.linalg.norm(pt1 - pt2) / np.sqrt(2)
                            side_lengths.append(dist)
            #找最小
            if side_lengths:
                min_side_length = min(side_lengths)
                all_squares.append({
                    'min_side_length': min_side_length,
                    'contour': cnt,
                    'area': area
                })
        if all_squares:
            # 找像素边长最小的
            smallest = min(all_squares, key=lambda s: s['min_side_length'])
            # 换算为真实尺寸
            current_size_mm = (smallest['min_side_length'] / paper_pixel_width) * KNOWN_A4_PAPER_WIDTH_MM
            current_size_cm = current_size_mm / 10.0
            size_to_send = current_size_cm
    # --- 步骤 4: 发送数据和显示结果 ---
    # 无论是否找到，都发送当前数据
    # distance_D_cm = 1.0285 * distance_D_cm - 7.9295 # PnP方法通常不需要线性校正，先注释掉
    send_data(ser_sender, 't2', 'txt',distance_D_cm)
    send_data(ser_sender, 't3', 'txt', size_to_send)

    # 在左上角显示距离D和最终要发送的边长x
    cv2.putText(frame, f"D (PnP): {distance_D_cm:.2f} cm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"x_min: {size_to_send:.2f} cm", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Task 5 - Find Smallest Square", frame)

    # 检查按键，如果按下 'q' 则返回 False 以退出程序
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True
