import cv2
import numpy as np
from utils.camera_converter import CameraConverter 
from utils.measurement import calculate_distance_cm, calculate_real_size_cm
from utils.serial_utils import send_data


# 1. 相机参数
converter = CameraConverter()
converter.current_camera = '2_1080'  
config = converter.get_camera_config()

# 将获取到的相机内参矩阵
CAMERA_MATRIX = config['mtx']
# 将获取到的相机畸变系数
DIST_COEFFS = config['dist']

# --- 2. 参考物真实尺寸 (单位: 毫米) ---
# A4纸的真实宽度，单位毫米。
KNOWN_A4_PAPER_WIDTH_MM = 210.0
# A4纸的真实高度，单位毫米。
KNOWN_A4_PAPER_HEIGHT_MM = 297.0


ADAPTIVE_THRESH_BLOCK_SIZE = 21 
ADAPTIVE_THRESH_C = 8       

MIN_PAPER_AREA = 3000  # A4纸在最远处时的最小面积
MAX_PAPER_AREA = 30000 # A4纸在最近处时的最大面积 (PnP适用范围更广，适当调大)

mouse_coords = (0, 0)

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

def get_mouse_coords(event, x, y, flags, param):
    global mouse_coords
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_coords = (x, y)

def run_task1(frame, ser_hmi): # <--- 修改这里
    if frame is None:
        print("任务1: 接收到空帧")
        return

    # --- 新增：定义一个统一的显示缩放比例 ---
    display_scale = 1.0

    # --- 步骤 1: 图像预处理 ---
    h, w = frame.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX, DIST_COEFFS, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFFS, None, new_camera_mtx)
    x_roi, y_roi, w_roi, h_roi = roi
    frame = undistorted_frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
    
    # 提取图像中心区域作为ROI (按要求保留)
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

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    #_, binary_frame = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_frame = cv2.adaptiveThreshold(
        blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C
    )
    kernel = np.ones((3, 1), np.uint8)
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel, iterations=1)
    # --- 修改：放大二值图以供显示 ---
    display_binary = cv2.resize(binary_frame, (0,0), fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
    cv2.imshow('binary', display_binary)

    # --- 步骤 2: 测量距离 D ---
    distance_D_cm = -1.0 
    paper_area = 0 
    paper_contour = None
    paper_pixel_width = -1.0

    contours, _ = cv2.findContours(binary_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # PnP方法对轮廓面积的适应性更强，可以适当放宽筛选条件
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
                    [0, 0, 0],                         # 左上
                    [KNOWN_A4_PAPER_WIDTH_MM, 0, 0],            # 右上
                    [KNOWN_A4_PAPER_WIDTH_MM, KNOWN_A4_PAPER_HEIGHT_MM, 0], # 右下
                    [0, KNOWN_A4_PAPER_HEIGHT_MM, 0]            # 左下
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
                # 注意：这里的points需要用排序后的点来保证宽度计算的稳定性
                sides = sorted([np.linalg.norm(sorted_image_points[i] - sorted_image_points[(i + 1) % 4]) for i in range(4)])
                paper_pixel_width = (sides[0] + sides[1]) / 2
                # paper_pixel_height = (sides[2] + sides[3]) / 2 # 如果需要也可以计算高度

                # 绘制轮廓和角点
                cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2) 
                for i, point in enumerate(sorted_image_points):
                    cv2.circle(frame, tuple(map(int, point)), 5, (0, 255, 0), -1)
                    cv2.putText(frame, str(i), tuple(map(int, point)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    # --- 步骤 3: 测量图形尺寸 x ---
    size_to_send = -1.0
    found_shapes = []
    valid_contours_to_draw = [] # 新增：用于存储所有要绘制的有效轮廓

    if paper_contour is not None and distance_D_cm > 0 and paper_pixel_width > 0:
        # 1. 获取A4纸轮廓的最小正矩形边界框
        x, y, w, h = cv2.boundingRect(paper_contour)

        # 2. 根据边界框裁剪二值图，得到A4纸区域
        #    这可以确保我们只处理A4纸内部的内容
        inner_binary = binary_frame[y:y+h, x:x+w]
        
        # --- 在新窗口中显示这个被正确处理的内部区域 ---
        display_inner_binary = cv2.resize(inner_binary, (0,0), fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('inner_binary', display_inner_binary)

        shape_contours, hierarchy = cv2.findContours(inner_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is not None:
            for i, contour in enumerate(shape_contours):
                # 只处理没有父轮廓的轮廓（即最外层的轮廓）
                if hierarchy[0][i][3] == -1:
                    area = cv2.contourArea(contour)
                    if area < 20 or area > paper_area * 0.65: # 增加一个最小面积过滤
                        print(f"跳过轮廓，面积过小或过大: {area}")
                        continue

                    pixel_length_for_calc = 0
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

                    if len(approx) == 3: # 三角形
                        sides = [np.linalg.norm(approx[i] - approx[(i + 1) % 3]) for i in range(3)]
                        pixel_length_for_calc = np.mean(sides)
                    elif len(approx) == 4: # 四边形
                        sides = [np.linalg.norm(approx[i] - approx[(i + 1) % 4]) for i in range(4)]
                        pixel_length_for_calc = np.mean(sides)
                    elif len(approx) > 6 or len(approx) < 3:
                        (cx, cy), radius = cv2.minEnclosingCircle(contour)
                        pixel_length_for_calc = radius * 2

                    if pixel_length_for_calc > 0:
                        print(f"找到轮廓，轮廓面积: {area}, 轮廓点数: {len(approx)}, 计算的像素长度: {pixel_length_for_calc}")
                        current_size_mm = (pixel_length_for_calc / paper_pixel_width) * KNOWN_A4_PAPER_WIDTH_MM
                        current_size_cm = current_size_mm / 10.0
                        found_shapes.append({'size': current_size_cm, 'area': area})
                        
                        # 关键：将轮廓坐标转换回原始 frame 的坐标系
                        contour_in_frame = contour + (x, y)
                        # 将转换后的有效轮廓添加到待绘制列表
                        valid_contours_to_draw.append(contour_in_frame)
                        
                        # 获取在原始 frame 中的包围框以正确放置文字
                        (x_b, y_b, _, _) = cv2.boundingRect(contour_in_frame)
                        cv2.putText(frame, f"{current_size_cm:.1f}cm", (x_b, y_b - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

    # --- 修改：在循环结束后，一次性绘制所有找到的有效闭合区域 ---
    if valid_contours_to_draw:
        cv2.drawContours(frame, valid_contours_to_draw, -1, (42, 42, 165), 2)

    # --- 步骤 4: 发送数据和显示结果 ---
    if found_shapes:
        largest_shape = max(found_shapes, key=lambda s: s['area'])
        size_to_send = largest_shape['size']

    # distance_D_cm = 1.0285 * distance_D_cm - 7.9295 # PnP方法通常不需要线性校正，先注释掉
    send_data(ser_hmi, 't2', 'txt',distance_D_cm) 
    send_data(ser_hmi, 't3', 'txt', size_to_send) 
 
    # --- 修改：调整字体大小和位置以适应小窗口 ---
    # 在左上角显示距离D和边长x
    cv2.putText(frame, f"D (PnP): {distance_D_cm:.2f} cm", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, f"x: {size_to_send:.2f} cm", (5, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 在左下角显示鼠标实时坐标
    cv2.putText(frame, f"Coords: {mouse_coords[0]}, {mouse_coords[1]}", 
                (5, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

    # --- 修改：将最终要显示的图像放大 ---
    display_frame = cv2.resize(frame, (0,0), fx=display_scale, fy=display_scale, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Task 1", display_frame)
    
    cv2.setMouseCallback("Task 1", get_mouse_coords)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True
