import cv2
import numpy as np
from utils.measurement import calculate_distance_cm, calculate_real_size_cm
from utils.serial_utils import send_data
from utils.camera_converter import CameraConverter
import re

# 相机参数
converter = CameraConverter()
converter.current_camera = '2_640'  
config = converter.get_camera_config()
CAMERA_MATRIX = config['mtx']
DIST_COEFFS = config['dist']
FOCAL_LENGTH = (CAMERA_MATRIX[0, 0] + CAMERA_MATRIX[1, 1]) / 2
KNOWN_A4_PAPER_WIDTH_MM = 210.0
KNOWN_A4_PAPER_HEIGHT_MM = 297.0

# --- 2. 图像处理参数 ---
# 定义了用于识别A4纸的最小像素面积。如果一个轮廓的面积小于这个值，它就会被忽略。
# 这有助于过滤掉图像中的小噪点或无关物体。
MIN_PAPER_AREA = 20000 
# 定义了用于识别A4纸的最大像素面积。如果一个轮廓的面积大于这个值，它也会被忽略。
# 这有助于排除比A4纸大得多的物体，或者在摄像头离得极近时防止错误识别。
MAX_PAPER_AREA = 200000

# --- 3. 全局变量，用于从外部接收目标编号 ---
# 这是一个全局变量，用来存储我们要寻找的目标数字。
# 初始值为-1，表示还没有指定任何目标。
target_number = -1 

def set_target_number(number):
    """从外部为任务7设置目标编号"""
    # 声明我们将要修改的是全局变量 target_number，而不是创建一个新的局部变量。
    global target_number
    # 打印一条日志，方便调试时知道目标数字已经被设置。
    print(f"Task 7: 目标编号已设置为 {number}")
    # 将从外部（main.py）接收到的数字赋值给全局变量。
    target_number = number

# --- 2. 修改函数定义，增加 ocr 参数 ---
def run_task8(frame, ser_sender, ocr): 
    if frame is None or ocr is None:
        # 如果是OCR模型为空，特别打印一条错误信息。
        if ocr is None:
            print("错误: OCR模型未正确传入Task 7")
        # 返回True表示程序可以继续运行（处理下一帧或等待），而不是退出。
        return True

    distance_D_cm = -1.0
    size_to_send = -1.0
    
    # --- 步骤 1: 图像预处理和畸变校正 (同Task1) ---
    # 获取输入图像的高度(h)和宽度(w)。
    h, w = frame.shape[:2]
    
    # 使用OpenCV的函数，根据我们预先标定的相机内参(CAMERA_MATRIX)和畸变系数(DIST_COEFFS)，
    # 计算出一个用于校正畸变的新相机矩阵(new_camera_mtx)和感兴趣区域(roi)。
    # 这个新矩阵能确保校正后的图像内容不被裁剪。
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX, DIST_COEFFS, (w, h), 1, (w, h))
    
    # 调用undistort函数，使用旧的相机参数和新计算出的校正矩阵，对原始图像进行畸变校正。
    undistorted_frame = cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFFS, None, new_camera_mtx)
    
    # 获取感兴趣区域(roi)的坐标和尺寸。roi是校正后图像中完全无黑边的有效区域。
    x_roi, y_roi, w_roi, h_roi = roi
    # 对校正后的图像进行裁剪，只保留roi区域。这样可以去除因校正产生的无用黑边。
    # 后续所有处理都将基于这个裁剪过的、无畸变的图像进行。
    frame = undistorted_frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
    
    # 将处理过的彩色图像转换为灰度图像。因为颜色信息对于寻找轮廓和OCR识别通常不是必需的，
    # 转为灰度可以简化计算，减少处理时间。
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 对灰度图像进行高斯模糊。这可以平滑图像，有效去除随机噪声（如传感器噪声），
    # 为下一步的阈值化操作提供一个更干净的输入，防止产生大量由噪点导致的错误轮廓。
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # 对模糊后的图像进行自适应阈值处理，生成二值图像（只有黑白两色）。
    # 与简单的全局阈值不同，自适应阈值会根据像素邻域内的亮度来决定该像素是变黑还是变白。
    # 这对于处理光照不均匀的图像（比如一边亮一边暗）效果非常好。
    binary_frame = cv2.adaptiveThreshold(blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 151, 30)
    
    # 定义一个5x5的结构元素（可以想象成一个小刷子）。
    kernel = np.ones((5, 5), np.uint8)
    # 执行形态学闭运算(MORPH_CLOSE)。闭运算=先膨胀后腐蚀。
    # 它的主要作用是填充物体内部的小黑洞、连接断开的白色区域，使目标轮廓更加完整和连续。
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 显示处理后的二值图像。这是一个调试步骤，可以让你直观地看到预处理的效果好坏，
    # 方便你调整高斯模糊、自适应阈值、形态学运算的参数。
    cv2.imshow("Binary Frame", binary_frame)

    # --- 步骤 2: 定位A4纸并计算距离D (同Task1) ---
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paper_contour = None
    paper_pixel_width = -1.0 # 新增：用于存储A4纸像素宽度作为比例尺
    if contours:
        valid_contours = [c for c in contours if MIN_PAPER_AREA < cv2.contourArea(c) < MAX_PAPER_AREA]
        if valid_contours:
            paper_contour = max(valid_contours, key=cv2.contourArea)
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

    # --- 步骤 3: 在A4纸内寻找所有正方形和数字 ---
    numbered_squares = {} # 存储 {数字: 正方形信息}
    if paper_contour is not None and distance_D_cm > 0 and paper_pixel_width > 0:
        paper_area = cv2.contourArea(paper_contour)
        # 创建A4纸区域的掩码
        paper_mask = np.zeros_like(binary_frame)
        cv2.drawContours(paper_mask, [paper_contour], -1, 255, -1)
        
        # --- 核心修改：只将外轮廓及其内部像素粘贴到白布上 ---
        # 1. 创建一张纯白的画布给OCR
        ocr_input_image = np.full_like(frame, 255, dtype=np.uint8)
        
        # 2. 在A4纸区域内寻找所有轮廓及其层级关系
        masked_binary = cv2.bitwise_and(binary_frame, paper_mask)
        shape_contours, hierarchy = cv2.findContours(masked_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # 3. 创建一个只包含所有外轮廓（方块）的掩码
        outer_contours_mask = np.zeros_like(binary_frame)
        if hierarchy is not None:
            for i, contour in enumerate(shape_contours):
                # hierarchy[0][i][3] == -1 表示这是最外层的轮廓（即方块）
                if hierarchy[0][i][3] == -1:
                    # 将方块轮廓填充到掩码中
                    cv2.drawContours(outer_contours_mask, [contour], -1, 255, -1)

        # 4. 使用掩码从二值图中“抠出”所有方块和数字
        #    这一步的结果是，只有方块和数字区域的像素被保留，其他地方都是黑色
        final_objects = cv2.bitwise_and(masked_binary, outer_contours_mask)

        # 5. 将“抠出”的内容粘贴到纯白画布上
        #    首先，创建外轮廓的彩色图像，方便后续粘贴
        outer_contours_color = cv2.cvtColor(final_objects, cv2.COLOR_GRAY2BGR)
        #    然后，将抠出的内容（彩色格式）添加到白色画布上
        ocr_input_image = cv2.add(ocr_input_image, outer_contours_color)

        # --- 3.1 OCR识别 ---
        # 对我们刚刚创建的、纯净的图像进行OCR识别
        ocr_result = ocr.ocr(ocr_input_image, cls=False)
        
        # --- 3.2 寻找正方形 ---
        # 注意：寻找正方形的操作仍然使用之前找到的轮廓和层级信息
        found_squares = []
        if hierarchy is not None:
            for i, cnt in enumerate(shape_contours):
                # hierarchy[0][i][3] == -1 表示这是最外层的轮廓（即方块）
                if hierarchy[0][i][3] == -1:
                    area = cv2.contourArea(cnt)
                    # 面积过滤条件保持不变
                    if not (500 < area < paper_area * 0.9): continue
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                    if len(approx) == 4:
                        # --- 计算和保存正方形信息的逻辑保持不变 ---
                        
                        # 1. 计算中心点
                        M = cv2.moments(cnt)
                        if M["m00"] == 0: continue
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # 2. 计算真实尺寸
                        sides = [np.linalg.norm(approx[j] - approx[(j + 1) % 4]) for j in range(4)]
                        pixel_side_length = np.mean(sides)
                        real_size_mm = (pixel_side_length / paper_pixel_width) * KNOWN_A4_PAPER_WIDTH_MM
                        real_size_cm = real_size_mm / 10.0
                        
                        # 3. 保存结果
                        found_squares.append({'contour': cnt, 'center': (cx, cy), 'size': real_size_cm})
                        
                        # 4. 绘制轮廓
                        cv2.drawContours(frame, [cnt], -1, (42, 42, 165), 2)

        # --- 3.3 匹配数字和正方形 ---
        if ocr_result and ocr_result[0] and found_squares:
            # 遍历所有OCR识别到的文本行
            for line in ocr_result[0]:
                # 提取识别到的文本内容、置信度以及整个文本块的边界框
                text, _ = line[1]
                box = line[0]
                
                # 遍历文本块中找到的所有数字
                for match in re.finditer(r'\d+', text):
                    # 提取找到的数字字符串，并转换为整数
                    num = int(match.group(0))
                    
                    # --- 恢复为“最近中心点”匹配逻辑，以处理重叠情况 ---
                    # 1. 计算文本块的中心点
                    text_moments = cv2.moments(np.array(box, dtype=np.int32))
                    if text_moments["m00"] == 0: continue
                    text_cx = int(text_moments["m10"] / text_moments["m00"])
                    text_cy = int(text_moments["m01"] / text_moments["m00"])

                    # 2. 找到离这个文本中心点最近的正方形
                    min_dist = float('inf')
                    best_match_square = None
                    for square in found_squares:
                        # 计算文本中心与每个正方形中心的欧氏距离
                        dist = np.linalg.norm(np.array(square['center']) - np.array((text_cx, text_cy)))
                        # 如果当前距离更小，则更新最小距离和最佳匹配
                        if dist < min_dist:
                            min_dist = dist
                            best_match_square = square
                    
                    # 3. 如果找到了最佳匹配的正方形，则建立关系
                    if best_match_square:
                        numbered_squares[num] = best_match_square
                        # 在匹配的正方形中心标注识别到的数字
                        cv2.putText(frame, str(num), best_match_square['center'], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # --- 步骤 4: 发送数据和显示结果 ---
    if target_number != -1 and target_number in numbered_squares:
        target_square = numbered_squares[target_number]
        size_to_send = target_square['size']
        
        # 高亮显示目标正方形
        cv2.drawContours(frame, [target_square['contour']], -1, (0, 255, 0), 3)
        # 显示其尺寸
        (x_b, y_b, _, _) = cv2.boundingRect(target_square['contour'])
        cv2.putText(frame, f"{size_to_send:.2f}cm", (x_b, y_b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 任务7 发送距离和边长
    send_data(ser_sender, distance_D_cm, size_to_send)

    # 在左上角显示信息
    cv2.putText(frame, f"Task: 7", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Target Num: {target_number}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"x: {size_to_send:.2f} cm", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Task 7 - Find Numbered Square", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True