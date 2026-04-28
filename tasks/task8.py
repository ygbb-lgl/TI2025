import cv2
import numpy as np
from compute import compute_homography, invert_homography
from scipy.optimize import least_squares
from utils.serial_utils import send_data
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

MIN_PAPER_AREA = 3000
MAX_PAPER_AREA = 30000


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

def run_task8(frame, ser_sender, ocr=None):
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
    scale_factor = 2
    frame = cv2.resize(frame, (int(w_roi * scale_factor), int(h_roi * scale_factor)), interpolation=cv2.INTER_LINEAR)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    _, binary_frame = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
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
        cv2.imshow("Task 8", frame)
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
    cv2.imshow('masked_binary', masked_binary)
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
            found_squares.append({'contour': cnt, 'size': real_side_cm})
            send_data(ser_sender, 't3', 'txt', real_side_cm)
            # 在正方形中心显示边长
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(frame, f"{real_side_cm:.2f}cm", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- 步骤5：显示 ---
    cv2.putText(frame, f"Task: 8", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # 缩小显示，适配高分辨率
    display_scale = 0.5
    display_frame = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
    cv2.imshow("Task 8 - Square Size", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True