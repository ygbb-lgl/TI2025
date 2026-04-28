def calculate_distance_cm(known_width_mm, focal_length_px, pixel_width):
    """
    根据相似三角形原理计算物体到摄像头的距离。

    Args:
        known_width_mm (float): 物体在真实世界中的已知宽度（单位：毫米）。
        focal_length_px (float): 摄像头的焦距（单位：像素）。
        pixel_width (float): 物体在图像中的宽度（单位：像素）。

    Returns:
        float: 计算出的距离（单位：厘米）。如果输入无效则返回-1.0。
    """
    if pixel_width <= 0:
        return -1.0
    # 公式: 距离 = (已知物体真实宽度 * 焦距) / 物体在图像中的像素宽度
    distance_mm = (known_width_mm * focal_length_px) / pixel_width
    return distance_mm / 10.0 # 转换为厘米

def calculate_real_size_cm(pixel_width, distance_cm, focal_length_px):
    """
    根据已知的距离，反向计算物体的真实尺寸。

    Args:
        pixel_width (float): 物体在图像中的宽度（单位：像素）。
        distance_cm (float): 物体已知的距离（单位：厘米）。
        focal_length_px (float): 摄像头的焦距（单位：像素）。

    Returns:
        float: 计算出的物体真实尺寸（单位：厘米）。如果输入无效则返回-1.0。
    """
    if distance_cm <= 0:
        return -1.0
    distance_mm = distance_cm * 10.0
    # 公式: 真实尺寸 = (物体像素宽度 * 距离) / 焦距
    size_mm = (pixel_width * distance_mm) / focal_length_px
    return size_mm / 10.0 # 转换为厘米