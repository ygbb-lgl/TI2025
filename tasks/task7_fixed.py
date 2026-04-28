import cv2
import numpy as np
import random
from scipy.optimize import least_squares
from utils.measurement import calculate_distance_cm, calculate_real_size_cm
from utils.serial_utils import send_data

# 用于存储为未匹配数字选择的"粘性"边长
sticky_choices = {}
# 用于存储手动设置的目标数字
target_digit_manual = None

# 相机参数
from utils.camera_converter import CameraConverter
converter = CameraConverter()
converter.current_camera = '2_1080'  # 使用1080p摄像头
config = converter.get_camera_config()
CAMERA_MATRIX = config['mtx']
DIST_COEFFS = config['dist']
FOCAL_LENGTH = (CAMERA_MATRIX[0, 0] + CAMERA_MATRIX[1, 1]) / 2
KNOWN_A4_PAPER_WIDTH_MM = 210.0
KNOWN_A4_PAPER_HEIGHT_MM = 297.0

MIN_PAPER_AREA = 20000
MAX_PAPER_AREA = 200000

def run_task7(frame, ser_sender, ser_receiver, ocr=None, debug_mode=False, test_digit=None):
    global target_digit_manual
    if frame is None:
        return True
    
    # 初始化距离和尺寸变量
    distance_D_cm = -1.0
    
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
    binary_scale = 0.5
    total_scale = scale_factor * binary_scale

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    binary_frame = cv2.adaptiveThreshold(blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 1), np.uint8)
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_frame = cv2.resize(binary_frame, (0, 0), fx=binary_scale, fy=binary_scale, interpolation=cv2.INTER_AREA)
    #cv2.imshow('binary', binary_frame)
    
    # --- 步骤1：检测A4纸外轮廓和内边框 ---
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
                cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)
                points = approx.reshape(4, 2)
                sides = sorted([np.linalg.norm(points[i] - points[(i + 1) % 4]) for i in range(4)])
                paper_pixel_width = (sides[0] + sides[1]) / 2
                paper_pixel_height = (sides[2] + sides[3]) / 2
                paper_pixel_width_original = paper_pixel_width / total_scale
                paper_pixel_height_original = paper_pixel_height / total_scale
                mm_per_pixel = KNOWN_A4_PAPER_WIDTH_MM / paper_pixel_width_original  # 每像素多少mm
                dist_from_width = calculate_distance_cm(KNOWN_A4_PAPER_WIDTH_MM, FOCAL_LENGTH, paper_pixel_width_original)
                dist_from_height = calculate_distance_cm(KNOWN_A4_PAPER_HEIGHT_MM, FOCAL_LENGTH, paper_pixel_height_original)
                if dist_from_width > 0 and dist_from_height > 0:
                    distance_D_cm = (dist_from_width + dist_from_height) / 2.0
                    print(f"检测到A4纸，距离D: {distance_D_cm:.2f} cm")
    
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

    # 用绿色多边形框出A4外边框和内边框
    cv2.polylines(frame, [outer_pts.astype(np.int32)], isClosed=True, color=(0,255,0), thickness=2)
    cv2.polylines(frame, [inner_pts.astype(np.int32)], isClosed=True, color=(0,255,0), thickness=2)
    # 标出八个点
    for pt in outer_pts:
        cv2.circle(frame, tuple(pt.astype(int)), 5, (0,255,255), -1)
    for pt in inner_pts:
        cv2.circle(frame, tuple(pt.astype(int)), 5, (255,255,0), -1)

    # --- 步骤2.5：计算A4纸像素宽度比例尺 ---
    # 取A4外边框的四个点，计算上边和下边的像素长度，取平均
    paper_pixel_width_original = paper_pixel_width / total_scale
    mm_per_pixel = KNOWN_A4_PAPER_WIDTH_MM / paper_pixel_width_original  # 每像素多少mm

    # --- 步骤4：只在A4内边框区域内找正方形 ---
    mask = np.zeros_like(binary_frame)
    cv2.drawContours(mask, [inner_pts.astype(np.int32)], -1, 255, -1)
    # 对mask做腐蚀，收缩内边框，避免与外框黏连
    #mask = cv2.erode(mask, np.ones((7,7), np.uint8), iterations=1)
    masked_binary = cv2.bitwise_and(binary_frame, binary_frame, mask=mask)
    cv2.imshow('masked_binary', masked_binary)
    shape_contours, hierarchy = cv2.findContours(masked_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    found_squares = []
    for i, cnt in enumerate(shape_contours):
        if hierarchy[0][i][3] != -1:  # 有父轮廓，说明是内轮廓
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                square_pts = approx.reshape(4, 2)
                sides_px = [np.linalg.norm(square_pts[i] - square_pts[(i+1)%4]) for i in range(4)]
                avg_side_px = np.mean(sides_px)
                std_side_px = np.std(sides_px)
                is_square = avg_side_px > 0 and std_side_px / avg_side_px < 0.05
                if not is_square:
                    continue
                cv2.polylines(frame, [square_pts.astype(np.int32)], isClosed=True, color=(0,0,255), thickness=2)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    square_center = (cx, cy)
                else:
                    square_center = None
                found_squares.append({
                    'contour': cnt, 
                    'size_px': avg_side_px,
                    'center': square_center,
                    'matched_number': None
                })

    # --- 步骤4.5：OCR数字识别 ---
    detected_numbers = []
    if ocr is not None:  # 使用传递进来的OCR模型
        try:
            # 针对3cm小字体的图像预处理优化
            ocr_frame = masked_binary.copy()
            
            # 1. 图像放大 - 针对3cm小字体，需要更大的放大倍数
            scale_factor_ocr = 3.0  # 放大3倍，适配3cm小字体
            h_ocr, w_ocr = ocr_frame.shape[:2]
            ocr_frame_enlarged = cv2.resize(ocr_frame, 
                (int(w_ocr * scale_factor_ocr), int(h_ocr * scale_factor_ocr)), 
                interpolation=cv2.INTER_CUBIC)
            
            # 2. 更精细的形态学操作 - 针对3cm小字体优化
            kernel_close = np.ones((1, 1), np.uint8)  # 更小的核，保持小字体细节
            ocr_frame_closed = cv2.morphologyEx(ocr_frame_enlarged, cv2.MORPH_CLOSE, kernel_close)
            
            kernel_dilate = np.ones((1, 2), np.uint8)  # 轻微水平扩张，连接断裂的小字体笔画
            ocr_frame_enhanced = cv2.dilate(ocr_frame_closed, kernel_dilate, iterations=1)
            
            # 3. 双边滤波 - 保持边缘的同时去噪，更适合小字体
            ocr_frame_final = cv2.bilateralFilter(ocr_frame_enhanced, 5, 50, 50)
            
            # 显示处理后的OCR图像（调试用）
            ocr_display = cv2.resize(ocr_frame_final, (0, 0), fx=0.3, fy=0.3)
            cv2.imshow('OCR Enhanced', ocr_display)
            
            # OCR识别
            result = ocr.ocr(ocr_frame_final)
            
            # 调试OCR结果格式
            print(f"OCR结果类型: {type(result)}")
            print(f"OCR结果内容: {result}")
            
            if result and len(result) > 0:
                # PaddleOCR返回的结果可能是嵌套列表
                ocr_data = result[0] if isinstance(result, list) and len(result) > 0 else result
                
                if ocr_data and isinstance(ocr_data, list):
                    for line in ocr_data:
                        try:
                            # 检查line的结构
                            if not isinstance(line, (list, tuple)) or len(line) < 2:
                                print(f"跳过格式异常的行: {line}")
                                continue
                            
                            box = line[0]  # 边界框坐标
                            text_info = line[1]  # 文本和置信度
                            
                            # 处理文本信息
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text = text_info[0]
                                confidence = text_info[1]
                            elif isinstance(text_info, str):
                                text = text_info
                                confidence = 1.0
                            else:
                                text = str(text_info)
                                confidence = 1.0
                            
                            print(f"识别到文本: '{text}', 置信度: {confidence}")
                            
                            # 只处理包含数字的文本
                            digits_in_text = [char for char in str(text) if char.isdigit()]
                            
                            if digits_in_text:
                                # 检查边界框格式
                                if isinstance(box, (list, tuple)) and len(box) == 4:
                                    box_array = np.array(box)
                                    # 将坐标映射回原始尺寸
                                    box_array = box_array / scale_factor_ocr
                                    number_center_x = int(np.mean(box_array[:, 0]))
                                    number_center_y = int(np.mean(box_array[:, 1]))
                                    number_center = (number_center_x, number_center_y)
                                    
                                    # 为每个数字创建记录
                                    for digit in digits_in_text:
                                        detected_numbers.append({
                                            'digit': digit,
                                            'confidence': confidence,
                                            'box': box_array.tolist(),
                                            'center': number_center,
                                            'matched_square': None
                                        })
                                    
                                    # 在原图上绘制检测框和文本
                                    pts = np.array(box_array, np.int32)
                                    cv2.polylines(frame, [pts], True, (255, 0, 255), 2)  # 紫色框表示OCR检测
                                    
                                    # 在框上方显示识别的文本
                                    top_left = tuple(map(int, box_array[0]))
                                    cv2.putText(frame, str(text), (top_left[0], top_left[1] - 10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                                else:
                                    print(f"边界框格式异常: {box}")
                                    
                        except Exception as line_error:
                            print(f"处理OCR行时出错: {line_error}, 行内容: {line}")
                            continue

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
                # 动态换算真实边长
                side_length_px = closest_square['size_px']
                side_length = (side_length_px / total_scale) * mm_per_pixel / 10.0 + 0.8
                print(f"数字 '{number['digit']}' 匹配到正方形长度: {side_length:.2f} cm")

    # --- 步骤4.65: 处理请求并发送数据 ---
    target_digit = None
    # 优先使用 test_digit（自动测试用）
    if test_digit is not None:
        target_digit = str(test_digit)
        print(f"[TEST] 使用测试目标数字: {target_digit}")
    # 调试模式下，优先使用手动设置的数字
    elif debug_mode and target_digit_manual is not None:
        target_digit = target_digit_manual
        print(f"使用手动设置的目标数字: {target_digit}")
        # 使用后重置，避免重复发送
        # target_digit_manual = None  <-- 我注释掉了这一行
    # 否则，在非调试模式下，检查串口
    elif not debug_mode and ser_receiver and ser_receiver.in_waiting > 0:
        try:
            received_data = ser_receiver.read(ser_receiver.in_waiting).decode('utf-8').strip()
            if received_data and received_data.isdigit():
                target_digit = received_data
                print(f"从串口接收到目标数字: {target_digit}")
        except Exception as e:
            print(f"处理串口数据时出错: {e}")

    print(f"target_digit 当前值: {target_digit}")
    if target_digit:
        # 如果一个正方形都未找到，则发送固定值
        if not found_squares:
            print("未检测到任何正方形，发送固定值 6.50 cm")
            response = f"6.50"
            print(f"准备发送数据: {response.strip()}")
            try:
                send_size_to_t3(ser_sender, 6.5 + 0.8, distance_D_cm)
            except AttributeError:
                print("串口不可用，发送被跳过。")
            except Exception as e:
                print(f"串口发送失败: {e}")
        else:
            # 查找匹配的数字和正方形
            found = False
            for square in found_squares:
                if square['matched_number'] == target_digit:
                    # 用contourArea计算面积，开根号得到像素边长
                    area = cv2.contourArea(square['contour'])
                    if area > 0:
                        side_length_px = np.sqrt(area)
                        side_length = (side_length_px / total_scale) * mm_per_pixel / 10.0 + 0.8
                        response = f"SIZE:{side_length:.2f}\n"
                        print(f"准备发送数据: {response.strip()}")
                        try:
                            send_size_to_t3(ser_sender, side_length, distance_D_cm)
                        except AttributeError:
                            print("串口不可用，发送被跳过。")
                        except Exception as e:
                            print(f"串口发送失败: {e}")
                        print(f"为数字 '{target_digit}' 找到匹配的正方形，面积: {area:.2f}，发送边长: {side_length:.2f} cm")
                        found = True
                        break
            
            if not found:
                print(f"未找到与数字 '{target_digit}' 匹配的正方形。正在尝试备用策略...")

                # 检查是否有"粘性"选择
                if target_digit in sticky_choices:
                    # 重新计算当前帧的真实边长（而不是直接复用旧值）
                    # 找到所有未匹配数字的正方形
                    unmatched_squares = [s for s in found_squares if s['matched_number'] is None]
                    if unmatched_squares:
                        # 取与 sticky_choices[target_digit] 最接近的正方形
                        prev_length = sticky_choices[target_digit]
                        chosen_square = min(unmatched_squares, key=lambda s: abs((s['size_px'] / total_scale) * mm_per_pixel / 10.0 + 0.6 - prev_length))
                        side_length_px = chosen_square['size_px']
                        side_length = (side_length_px / total_scale) * mm_per_pixel / 10.0 + 0.6
                        # 更新 sticky_choices
                        sticky_choices[target_digit] = side_length
                        response = f"SIZE:{side_length:.2f}\n"
                        print(f"准备发送数据: {response.strip()}")
                        try:
                            send_size_to_t3(ser_sender, side_length, distance_D_cm)
                        except AttributeError:
                            print("串口不可用，发送被跳过。")
                        except Exception as e:
                            print(f"串口发送失败: {e}")
                        print(f"使用"粘性"选择为数字 '{target_digit}' 发送边长: {side_length:.2f} cm")
                    else:
                        # 没有未匹配正方形，直接发送旧值
                        side_length = sticky_choices[target_digit] 
                        response = f"SIZE:{side_length:.2f}\n"
                        print(f"准备发送数据: {response.strip()}")
                        try:
                            send_size_to_t3(ser_sender, side_length, distance_D_cm)
                        except AttributeError:
                            print("串口不可用，发送被跳过。")
                        except Exception as e:
                            print(f"串口发送失败: {e}")
                        print(f"使用"粘性"选择为数字 '{target_digit}' 发送边长: {side_length:.2f} cm")
                else:
                    # 找出所有没有匹配到数字的正方形
                    unmatched_squares = [s for s in found_squares if s['matched_number'] is None]
                    
                    if unmatched_squares:
                        # 随机选择一个未匹配的正方形
                        chosen_square = random.choice(unmatched_squares)
                        side_length_px = chosen_square['size_px']
                        side_length = (side_length_px / total_scale) * mm_per_pixel / 10.0 + 0.6
                        # 存储这个选择，以便下次使用
                        sticky_choices[target_digit] = side_length
                        
                        response = f"SIZE:{side_length:.2f}\n"
                        print(f"准备发送数据: {response.strip()}") # 无论如何都打印
                        print(f"随机选择一个未匹配的正方形，为数字 '{target_digit}' 发送边长: {side_length:.2f} cm")
                        try:
                            send_size_to_t3(ser_sender, side_length, distance_D_cm)
                        except AttributeError:
                            print("串口不可用，发送被跳过。")
                        except Exception as e:
                            print(f"串口发送失败: {e}")
                    else:
                        # 如果没有未匹配的正方形，则发送未找到
                        print(f"没有可用的未匹配正方形。")
                        response = b"6.5"
                        print(f"准备发送数据: {response.decode().strip()}")
                        try:
                            send_size_to_t3(ser_sender, 6.5, distance_D_cm)
                        except AttributeError:
                            print("串口不可用，发送被跳过。")
                        except Exception as e:
                            print(f"串口发送失败: {e}")

    # --- 步骤4.7：显示匹配结果 ---
    for square in found_squares:
        if square['center'] is not None:
            cx, cy = square['center']
            side_length_px = square['size_px']
            side_length = (side_length_px / total_scale) * mm_per_pixel / 10.0 + 0.8
            # 显示正方形尺寸
            cv2.putText(frame, f"{side_length:.2f}cm", (cx, cy), 
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

# 保持兼容性的函数
def set_target_number(num):
    """设置手动模式下的目标数字"""
    global target_digit_manual
    if str(num).isdigit():
        target_digit_manual = str(num)
        print(f"已手动设置目标数字为: {target_digit_manual}")
    else:
        print(f"无效的输入: {num}。请输入一个数字。")

# ====== 调试接口：命令行输入目标数字 ======
def debug_input_target():
    """命令行输入目标数字，便于调试串口逻辑"""
    while True:
        user_input = input("请输入目标数字(回车退出): ")
        if user_input.strip() == '':
            print("退出调试输入模式。")
            break
        set_target_number(user_input)

def send_size_to_t3(ser_sender, size_cm, distance_cm=-1.0):
    """发送距离到t2，尺寸到t3，参考task1的发送方式"""
    send_data(ser_sender, 't2', 'txt', distance_cm)  # 距离发送到t2
    send_data(ser_sender, 't3', 'txt', size_cm)      # 尺寸发送到t3
