
import cv2
import numpy as np
from utils.camera_converter import CameraConverter
import time

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

def pnp_distance_estimation(frame, camera_matrix, dist_coeffs, known_width_mm, known_height_mm):
    """
    使用 PnP 算法估计相机到目标的距离和姿态。

    Args:
        frame (np.array): 输入的视频帧。
        camera_matrix (np.array): 相机内参矩阵。
        dist_coeffs (np.array): 相机畸变系数。
        known_width_mm (float): 目标物体的实际宽度（毫米）。
        known_height_mm (float): 目标物体的实际高度（毫米）。

    Returns:
        tuple: (处理后的图像, 距离(cm))
    """
    # --- 1. 图像预处理 (仿照 task1.py) ---
    # 图像去畸变
    h, w = frame.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)
    x_roi, y_roi, w_roi, h_roi = roi
    frame = undistorted_frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

    # 提取中心ROI (如果需要可以调整)
    h, w = frame.shape[:2]
    roi_x_start = 860
    roi_x_end = w - roi_x_start
    roi_y_start = 350
    roi_y_end = h - roi_y_start
    if roi_x_start < roi_x_end and roi_y_start < roi_y_end:
        frame = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    # 转换为灰度图并进行二值化
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    # 使用自适应阈值来应对不同光照条件
    binary_frame = cv2.adaptiveThreshold(
        blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 8
    )
    # 形态学开运算，去除小的噪声点
    kernel = np.ones((3, 1), np.uint8)
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 在一个新窗口显示二值化结果，方便调试
    cv2.imshow("Binary for PnP", binary_frame)

    # --- 2. 寻找A4纸的轮廓和角点 ---
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    distance_cm = -1.0

    if contours:
        # 假设最大的轮廓是我们的目标A4纸
        paper_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(paper_contour)

        # 增加一个面积筛选，避免识别到过小或过大的物体
        if 3000 < area < 500000: # 这个范围可能需要根据实际情况调整
            peri = cv2.arcLength(paper_contour, True)
            # 多边形逼近，找到轮廓的四个角点
            approx = cv2.approxPolyDP(paper_contour, 0.02 * peri, True)

            if len(approx) == 4:
                # --- 3. PnP 求解 ---
                # 绘制A4纸轮廓
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                
                # 3.1 定义世界坐标系中的3D点 (A4纸的四个角点)
                # 我们将A4纸的左上角作为坐标原点 (0,0,0)
                object_points = np.array([
                    [0, 0, 0],                         # 左上
                    [known_width_mm, 0, 0],            # 右上
                    [known_width_mm, known_height_mm, 0], # 右下
                    [0, known_height_mm, 0]            # 左下
                ], dtype="float32")

                # 3.2 获取图像坐标系中的2D点，并排序
                image_points = approx.reshape(4, 2).astype('float32')
                sorted_image_points = order_points(image_points)

                # 绘制排序后的角点，方便观察
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)] # B, G, R, C
                labels = ["TL", "TR", "BR", "BL"]
                for i in range(4):
                    cv2.circle(frame, tuple(map(int, sorted_image_points[i])), 5, colors[i], -1)
                    cv2.putText(frame, labels[i], tuple(map(int, sorted_image_points[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                # 3.3 调用 solvePnP
                success, rvec, tvec = cv2.solvePnP(object_points, sorted_image_points, camera_matrix, dist_coeffs)

                if success:
                    # tvec 是相机坐标系中，目标物体相对于相机的位置
                    # tvec[2] 就是我们需要的深度信息（距离）
                    distance_mm = tvec[2][0]
                    distance_cm = distance_mm / 10.0
                    distance_cm = distance_cm - 5

                    # 3.4 可选：在图像上绘制坐标轴以可视化姿态
                    axis_points = np.float32([[0,0,0], [50,0,0], [0,50,0], [0,0,-50]]).reshape(-1,3)
                    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
                    
                    # 绘制坐标轴
                    origin = tuple(map(int, imgpts[0].ravel()))
                    cv2.line(frame, origin, tuple(map(int, imgpts[1].ravel())), (255,0,0), 3) # X轴: 蓝色
                    cv2.line(frame, origin, tuple(map(int, imgpts[2].ravel())), (0,255,0), 3) # Y轴: 绿色
                    cv2.line(frame, origin, tuple(map(int, imgpts[3].ravel())), (0,0,255), 3) # Z轴: 红色

    # 在图像左上角显示计算出的距离
    cv2.putText(frame, f"Distance: {distance_cm:.2f} cm", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    print(distance_cm)
    
    return frame, distance_cm

if __name__ == '__main__':
    # --- 1. 初始化相机参数 ---
    converter = CameraConverter()
    converter.current_camera = '2_1080'  # 根据您的相机选择配置
    config = converter.get_camera_config()
    CAMERA_MATRIX = config['mtx']
    DIST_COEFFS = config['dist']

    # --- 2. A4纸的真实尺寸 (单位: 毫米) ---
    KNOWN_A4_PAPER_WIDTH_MM = 210.0
    KNOWN_A4_PAPER_HEIGHT_MM = 297.0

    # --- 3. 打开摄像头 ---
    cap = cv2.VideoCapture(0) # 0 代表默认摄像头，如果需要请修改
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        exit()
        
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    prev_time = 0

    # --- 4. 主循环 ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取视频帧")
            break

        # 计算并显示帧率
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 调用PnP测距函数
        processed_frame, distance = pnp_distance_estimation(
            frame, CAMERA_MATRIX, DIST_COEFFS, 
            KNOWN_A4_PAPER_WIDTH_MM, KNOWN_A4_PAPER_HEIGHT_MM
        )
        
        # 显示结果图像
        # 原始图像可能太大，可以缩小显示
        display_frame = cv2.resize(processed_frame, (0,0), fx=0.5, fy=0.5)
        cv2.imshow("PnP Distance Estimation", display_frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 5. 释放资源 ---
    cap.release()
    cv2.destroyAllWindows()
