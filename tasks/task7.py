import cv2
import numpy as np
from compute import compute_homography, invert_homography
from scipy.optimize import least_squares

# OCR支持
try:
    from paddleocr import PaddleOCR
    # 初始化OCR模型（全局初始化，避免重复创建）
    _ocr_model = None
    def get_ocr_model():
        global _ocr_model
        if _ocr_model is None:
            print("正在初始化OCR模型...")
            _ocr_model = PaddleOCR(use_angle_cls=True, lang='ch')
            print("OCR模型初始化完成")
        return _ocr_model
    OCR_AVAILABLE = True
except ImportError:
    print("警告: PaddleOCR未安装，任务7将无法使用OCR功能")
    def get_ocr_model():
        return None
    OCR_AVAILABLE = False

# 相机参数
from utils.camera_converter import CameraConverter
converter = CameraConverter()
converter.current_camera = '2_1080'  # 使用1080p摄像头
config = converter.get_camera_config()
CAMERA_MATRIX = config['mtx']
DIST_COEFFS = config['dist']

KNOWN_A4_PAPER_WIDTH_MM = 210.0
KNOWN_A4_PAPER_HEIGHT_MM = 297.0
BORDER_WIDTH_MM = 20.0  # 2cm = 20mm

MIN_PAPER_AREA = 20000
MAX_PAPER_AREA = 200000


def project_points(world_points, H):
    """用单应性矩阵H将世界点投影到像素坐标系"""
    world_h = np.hstack([world_points, np.ones((world_points.shape[0], 1))])
    img_proj = (H @ world_h.T).T
    img_proj = img_proj[:, :2] / img_proj[:, 2:3]
    return img_proj

def homography_residuals(h, world_points, image_points):
    """重投影误差"""
    H = h.reshape(3, 3)
    proj = project_points(world_points, H)
    return (proj - image_points).ravel()

def optimize_homography_LM(world_points, image_points, H_init=None):
    """
    用LM算法优化单应性矩阵H
    :param world_points: Nx2 世界坐标
    :param image_points: Nx2 像素坐标
    :param H_init: 初始H（3x3），可用DLT结果
    :return: 优化后的H（3x3）
    """
    if H_init is None:
        from compute import compute_homography
        H_init = compute_homography(world_points, image_points)
    h0 = H_init.flatten() / H_init[2,2]
    res = least_squares(
        homography_residuals, h0, method='lm',
        args=(world_points, image_points)
    )
    H_optimized = res.x.reshape(3, 3)
    H_optimized = H_optimized / H_optimized[2, 2]
    return H_optimized

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # 左上
    rect[2] = pts[np.argmax(s)]      # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # 右上
    rect[3] = pts[np.argmax(diff)]   # 左下
    return rect

def run_task7(frame, ser_sender, ocr=None):
    if frame is None:
        return True
    # 图像预处理，先去畸变
    h, w = frame.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX, DIST_COEFFS, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFFS, None, new_camera_mtx)
    x_roi, y_roi, w_roi, h_roi = roi
    frame = undistorted_frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

    # 提取中心区域ROI（参考task1）
    h, w = frame.shape[:2]
    roi_x_start = 600
    roi_x_end   = w - roi_x_start
    roi_y_start = 350
    roi_y_end   = h - roi_y_start
    frame = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    # 对ROI裁剪后的图像进行放大（如3倍）
    h_roi, w_roi = frame.shape[:2]
    scale_factor = 4
    frame = cv2.resize(frame, (int(w_roi * scale_factor), int(h_roi * scale_factor)), interpolation=cv2.INTER_LINEAR)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    binary_frame = cv2.adaptiveThreshold(blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 1), np.uint8)
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_scale = 0.5
    binary_frame = cv2.resize(binary_frame, (0, 0), fx=binary_scale, fy=binary_scale, interpolation=cv2.INTER_AREA)
    cv2.imshow('binary', binary_frame)
    # --- 步骤1：检测A4纸外轮廓和内边框 ---
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            squares.append((area, approx))
    # 按面积排序，最大为外边框，次大为内边框
    if len(squares) >= 2:
        squares = sorted(squares, key=lambda x: -x[0])
        outer_pts = squares[0][1].reshape(4, 2)
        inner_pts = squares[1][1].reshape(4, 2)
    else:
        outer_pts = None
        inner_pts = None

    if outer_pts is None or inner_pts is None:
        cv2.imshow("Task 7", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True

    outer_pts = order_points(outer_pts)
    inner_pts = order_points(inner_pts)

    # 用绿色多边形框出A4外边框和内边框
    cv2.polylines(frame, [outer_pts.astype(np.int32)], isClosed=True, color=(0,255,0), thickness=2)
    cv2.polylines(frame, [inner_pts.astype(np.int32)], isClosed=True, color=(0,255,0), thickness=2)
    # 标出八个点
    for pt in outer_pts:
        cv2.circle(frame, tuple(pt.astype(int)), 5, (0,255,255), -1)
    for pt in inner_pts:
        cv2.circle(frame, tuple(pt.astype(int)), 5, (255,255,0), -1)

    # --- 步骤2：构造八个点的世界坐标和像素坐标 ---
    # 世界坐标（单位mm），以A4纸中心为原点
    half_w = KNOWN_A4_PAPER_WIDTH_MM / 2
    half_h = KNOWN_A4_PAPER_HEIGHT_MM / 2
    half_w_in = half_w - BORDER_WIDTH_MM
    half_h_in = half_h - BORDER_WIDTH_MM
    world_outer = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h]
    ], dtype=np.float32)
    world_inner = np.array([
        [-half_w_in, -half_h_in],
        [ half_w_in, -half_h_in],
        [ half_w_in,  half_h_in],
        [-half_w_in,  half_h_in]
    ], dtype=np.float32)
    world_points = np.vstack([world_outer, world_inner])
    image_points = np.vstack([outer_pts, inner_pts])

    # --- 步骤3：计算单应性矩阵 ---
    H_init = compute_homography(world_points, image_points)
    H = optimize_homography_LM(world_points, image_points, H_init=H_init)
    
    # --- 步骤4：只在A4内边框区域内找正方形 ---
    mask = np.zeros_like(binary_frame)
    cv2.drawContours(mask, [inner_pts.astype(np.int32)], -1, 255, -1)
    # 对mask做腐蚀，收缩内边框，避免与外框黏连
    mask = cv2.erode(mask, np.ones((7,7), np.uint8), iterations=1)
    masked_binary = cv2.bitwise_and(binary_frame, binary_frame, mask=mask)
    #cv2.imshow('masked_binary', masked_binary)
    shape_contours, _ = cv2.findContours(masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_squares = []
    for cnt in shape_contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4:
            square_pts = approx.reshape(4, 2)
            # 用红色画正方形
            cv2.polylines(frame, [square_pts.astype(np.int32)], isClosed=True, color=(0,0,255), thickness=2)
            # 利用H矩阵反解世界坐标
            world_square = invert_homography(square_pts, H)
            print("反解出的正方形世界坐标：")
            for idx, pt in enumerate(world_square):
                print(f"顶点{idx+1}: ({pt[0]:.2f}, {pt[1]:.2f})")
            # 计算四边长，取平均
            sides = [np.linalg.norm(world_square[i] - world_square[(i+1)%4]) for i in range(4)]
            real_side_cm = np.mean(sides) / 10.0 + 0.2  # 加0.2cm的误差补偿
            
            # 计算正方形中心点
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                square_center = (cx, cy)
            else:
                square_center = None
            
            found_squares.append({
                'contour': cnt, 
                'size': real_side_cm,
                'center': square_center,
                'matched_number': None  # 用于存储匹配的数字
            })

    # --- 步骤4.5：OCR数字识别 ---
    detected_numbers = []
    if OCR_AVAILABLE:
        try:
            ocr = get_ocr_model()
            if ocr is not None:
                # 对当前帧进行OCR识别
                # 可以选择对原始帧或处理后的帧进行识别
                ocr_frame = masked_binary.copy()
                
                # OCR识别
                result = ocr.ocr(ocr_frame)
                
                if result and result[0]:
                    for line in result[0]:
                        if len(line) >= 2:
                            box = line[0]  # 边界框坐标
                            text_info = line[1]  # 文本和置信度
                            
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text = text_info[0]
                                confidence = text_info[1]
                            else:
                                text = str(text_info)
                                confidence = 1.0
                            
                            # 只处理数字
                            for char in text:
                                if char.isdigit():
                                    # 计算数字的中心点
                                    if len(box) == 4:
                                        box_array = np.array(box)
                                        number_center_x = int(np.mean(box_array[:, 0]))
                                        number_center_y = int(np.mean(box_array[:, 1]))
                                        number_center = (number_center_x, number_center_y)
                                    else:
                                        number_center = None
                                    
                                    detected_numbers.append({
                                        'digit': char,
                                        'confidence': confidence,
                                        'box': box,
                                        'center': number_center,
                                        'matched_square': None  # 用于存储匹配的正方形
                                    })
                            
                            # 在原图上绘制检测框和文本（如果包含数字）
                            if any(c.isdigit() for c in text) and len(box) == 4:
                                pts = np.array(box, np.int32)
                                cv2.polylines(frame, [pts], True, (255, 0, 255), 2)  # 紫色框表示OCR检测
                                
                                # 在框上方显示识别的文本
                                top_left = tuple(map(int, box[0]))
                                cv2.putText(frame, text, (top_left[0], top_left[1] - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        except Exception as e:
            print(f"OCR处理出错: {e}")
            cv2.putText(frame, f"OCR Error", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- 步骤4.6：匹配正方形和数字 ---
    if found_squares and detected_numbers:
        # 为每个数字找到最近的正方形
        for number in detected_numbers:
            if number['center'] is None:
                continue
                
            min_distance = float('inf')
            closest_square = None
            
            for square in found_squares:
                if square['center'] is None:
                    continue
                
                # 计算数字中心和正方形中心的欧氏距离
                distance = np.sqrt((number['center'][0] - square['center'][0])**2 + 
                                 (number['center'][1] - square['center'][1])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_square = square
            
            # 建立匹配关系
            if closest_square is not None:
                number['matched_square'] = closest_square
                closest_square['matched_number'] = number['digit']
                print(f"数字 '{number['digit']}' 匹配到正方形，距离: {min_distance:.2f} 像素")

    # --- 步骤4.7：显示匹配结果 ---
    for square in found_squares:
        if square['center'] is not None:
            cx, cy = square['center']
            
            # 显示正方形尺寸
            cv2.putText(frame, f"{square['size']:.2f}cm", (cx, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 如果有匹配的数字，显示数字信息
            if square['matched_number'] is not None:
                cv2.putText(frame, f"Num: {square['matched_number']}", (cx, cy + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # 绘制正方形中心点
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    
    # 显示数字中心点
    for number in detected_numbers:
        if number['center'] is not None:
            nx, ny = number['center']
            cv2.circle(frame, (nx, ny), 3, (255, 0, 255), -1)  # 紫色圆点表示数字中心

    # 显示检测到的数字信息
    if detected_numbers:
        numbers_text = ', '.join([d['digit'] for d in detected_numbers])
        cv2.putText(frame, f"Numbers: {numbers_text}", (20, 70), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        print(f"检测到的数字: {numbers_text}")

    # --- 步骤5：显示 ---
    cv2.putText(frame, f"Task: 7", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # 缩小显示，适配高分辨率
    display_scale = 0.5
    display_frame = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
    cv2.imshow("Task 7 - Square Size + OCR", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True

# 保持兼容性的函数（现在不使用）
def set_target_number(num):
    """设置目标数字（保持接口兼容性，但现在只进行数字识别）"""
    print(f"注意: 任务7现在只进行数字识别，不需要设置目标数字。输入的数字{num}已忽略。")
