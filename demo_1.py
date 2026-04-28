import cv2
import numpy as np
import os

def detect_black_edges_and_save_steps():
    # 设置输入图像的绝对路径 - 请将路径替换为您的图片实际路径
    image_path = r"C:\Users\lgl20\Desktop\TI\2025\5f929db38a7c880b3b17aefb6437d633.jpg"  # 请修改这里的路径
    
    # 设置输出目录
    output_dir = r"C:\Users\lgl20\Desktop\TI\2025"  # 请修改这里的路径
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取图像
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # --- 步骤 1: 转换为灰度图 ---
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_path = os.path.join(output_dir, "1_gray_image.jpg")
    cv2.imwrite(gray_path, gray_frame)
    print(f"已保存灰度图: {gray_path}")
    
    # --- 步骤 2: 高斯滤波 ---
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    blurred_path = os.path.join(output_dir, "2_gaussian_blur.jpg")
    cv2.imwrite(blurred_path, blurred_frame)
    print(f"已保存高斯滤波: {blurred_path}")
    
    # --- 步骤 3: 自适应二值化 ---
    binary_frame = cv2.adaptiveThreshold(
        blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 151, 30
    )
    
    # 形态学闭运算连接边缘
    kernel = np.ones((5, 5), np.uint8)
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    binary_path = os.path.join(output_dir, "3_adaptive_binary.jpg")
    cv2.imwrite(binary_path, binary_frame)
    print(f"已保存自适应二值化: {binary_path}")
    
    # --- 步骤 4: 边缘检测 ---
    # 使用Canny边缘检测
    edges = cv2.Canny(blurred_frame, 50, 150)
    edges_path = os.path.join(output_dir, "4_edge_detection.jpg")
    cv2.imwrite(edges_path, edges)
    print(f"已保存边缘检测: {edges_path}")
    
    # --- 步骤 5: 轮廓检测 ---
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建轮廓检测结果图像
    contour_frame = frame.copy()
    
    # 绘制所有检测到的轮廓
    for contour in contours:
        # 过滤掉面积过小的轮廓（噪声）
        area = cv2.contourArea(contour)
        if area < 100:  # 最小面积阈值
            continue
            
        # 绘制轮廓（红色）
        cv2.drawContours(contour_frame, [contour], -1, (0, 0, 255), 2)
        
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 绘制边界框（绿色）
        cv2.rectangle(contour_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 显示轮廓面积
        cv2.putText(contour_frame, f"Area: {area:.0f}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 显示检测到的轮廓数量
    cv2.putText(contour_frame, f"Contours: {len(contours)}", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    contour_path = os.path.join(output_dir, "5_detected_contours.jpg")
    cv2.imwrite(contour_path, contour_frame)
    print(f"已保存轮廓检测结果: {contour_path}")
    
    # --- 步骤 6: Shi-Tomasi角点检测 ---
    # 转换为彩色图像用于绘制特征点
    corners_frame = frame.copy()
    
    # Shi-Tomasi角点检测参数
    max_corners = 100
    quality_level = 0.01
    min_distance = 10
    
    # 检测角点
    corners = cv2.goodFeaturesToTrack(gray_frame, max_corners, quality_level, min_distance)
    
    if corners is not None:
        corners = np.int0(corners)
        
        # 绘制角点
        for i, corner in enumerate(corners):
            x, y = corner.ravel()
            # 绘制绿色圆点
            cv2.circle(corners_frame, (x, y), 5, (0, 255, 0), -1)
            # 显示角点编号
            cv2.putText(corners_frame, str(i+1), (x+5, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # 显示检测到的角点数量
    corner_count = len(corners) if corners is not None else 0
    cv2.putText(corners_frame, f"Shi-Tomasi Corners: {corner_count}", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    corners_path = os.path.join(output_dir, "6_shi_tomasi_corners.jpg")
    cv2.imwrite(corners_path, corners_frame)
    print(f"已保存Shi-Tomasi角点检测结果: {corners_path}")
    
    # --- 步骤 7: Harris角点检测 ---
    harris_frame = frame.copy()
    
    # Harris角点检测参数
    block_size = 2
    aperture_size = 3
    k = 0.04
    
    # Harris角点检测
    harris_dst = cv2.cornerHarris(gray_frame, block_size, aperture_size, k)
    
    # 膨胀以标记角点
    harris_dst = cv2.dilate(harris_dst, None)
    
    # 设置阈值，标记角点
    threshold = 0.01 * harris_dst.max()
    harris_corner_count = 0
    
    for i in range(harris_dst.shape[0]):
        for j in range(harris_dst.shape[1]):
            if harris_dst[i, j] > threshold:
                # 绘制红色圆点
                cv2.circle(harris_frame, (j, i), 3, (0, 0, 255), -1)
                harris_corner_count += 1
    
    # 显示检测到的角点数量
    cv2.putText(harris_frame, f"Harris Corners: {harris_corner_count}", (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    harris_path = os.path.join(output_dir, "7_harris_corners.jpg")
    cv2.imwrite(harris_path, harris_frame)
    print(f"已保存Harris角点检测结果: {harris_path}")
    
    # --- 步骤 8: FAST特征点检测 ---
    fast_frame = frame.copy()
    
    # 创建FAST检测器
    fast = cv2.FastFeatureDetector_create()
    
    # 检测关键点
    kp = fast.detect(gray_frame, None)
    
    # 绘制关键点
    fast_frame = cv2.drawKeypoints(fast_frame, kp, None, color=(255, 0, 0), flags=0)
    
    # 显示检测到的特征点数量
    cv2.putText(fast_frame, f"FAST Features: {len(kp)}", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    fast_path = os.path.join(output_dir, "8_fast_features.jpg")
    cv2.imwrite(fast_path, fast_frame)
    print(f"已保存FAST特征点检测结果: {fast_path}")
    
    # --- 步骤 9: 综合结果显示（轮廓+Shi-Tomasi角点）---
    combined_frame = frame.copy()
    
    # 绘制轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        cv2.drawContours(combined_frame, [contour], -1, (0, 0, 255), 2)
        
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(combined_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 绘制Shi-Tomasi角点
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            # 绘制黄色大圆点
            cv2.circle(combined_frame, (x, y), 8, (0, 255, 255), -1)
            # 绘制蓝色小圆点
            cv2.circle(combined_frame, (x, y), 3, (255, 0, 0), -1)
    
    # 添加图例
    cv2.putText(combined_frame, "Red: Contours", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(combined_frame, "Green: Bounding Box", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(combined_frame, "Yellow: Feature Points", (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(combined_frame, f"Contours: {len(contours)} | Corners: {corner_count}", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    combined_path = os.path.join(output_dir, "9_combined_result.jpg")
    cv2.imwrite(combined_path, combined_frame)
    print(f"已保存综合结果: {combined_path}")
    
    # --- 步骤 10: 边缘特征点检测 ---
    # 在边缘图像上检测角点
    edge_corners_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 在边缘图像上检测角点
    edge_corners = cv2.goodFeaturesToTrack(edges, 50, 0.1, 20)
    
    if edge_corners is not None:
        edge_corners = np.int0(edge_corners)
        for corner in edge_corners:
            x, y = corner.ravel()
            cv2.circle(edge_corners_frame, (x, y), 5, (0, 0, 255), -1)
    
    cv2.putText(edge_corners_frame, f"Edge Corners: {len(edge_corners) if edge_corners is not None else 0}", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    edge_corners_path = os.path.join(output_dir, "10_edge_corners.jpg")
    cv2.imwrite(edge_corners_path, edge_corners_frame)
    print(f"已保存边缘角点检测结果: {edge_corners_path}")
    
    print(f"\n检测统计:")
    print(f"  - 轮廓数量: {len(contours)}")
    print(f"  - Shi-Tomasi角点数量: {corner_count}")
    print(f"  - Harris角点数量: {harris_corner_count}")
    print(f"  - FAST特征点数量: {len(kp)}")
    
    # 显示所有处理结果
    cv2.imshow('1. Gray Image', gray_frame)
    cv2.imshow('2. Gaussian Blur', blurred_frame)
    cv2.imshow('3. Adaptive Binary', binary_frame)
    cv2.imshow('4. Edge Detection', edges)
    cv2.imshow('5. Detected Contours', contour_frame)
    cv2.imshow('6. Shi-Tomasi Corners', corners_frame)
    cv2.imshow('7. Harris Corners', harris_frame)
    cv2.imshow('8. FAST Features', fast_frame)
    cv2.imshow('9. Combined Result', combined_frame)
    cv2.imshow('10. Edge Corners', edge_corners_frame)
    
    print(f"\n所有处理结果已保存在: {output_dir}")
    
    # 等待按键关闭窗口
    print("\n按任意键关闭所有窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 主函数
def main():
    # 修改这两行路径为您实际的路径
    input_image_path = r"C:\your\absolute\path\to\your\image.jpg"  # 输入图片路径
    output_directory = r"C:\output\images\directory"  # 输出目录
    
    # 如果您有具体的图片，请取消下面两行的注释并修改路径
    # 然后注释掉下面的 detect_black_edges_and_save_steps() 调用
    
    # 示例：假设您的图片是 test_image.png
    # input_image_path = r"C:\Users\YourName\Desktop\test_image.png"
    # output_directory = r"C:\Users\YourName\Desktop\processed_images"
    
    detect_black_edges_and_save_steps()

if __name__ == "__main__":
    main()