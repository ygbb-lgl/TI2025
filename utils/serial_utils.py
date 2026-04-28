import serial

def send_data(ser_sender, distance_cm, size_x_cm):
    if not (ser_sender and ser_sender.is_open):
        return

    # 定义串口屏指令的结束符
    end_cmd = bytes.fromhex('ff ff ff')

    try:
        # 1. 准备要发送的距离和尺寸字符串
        # 如果值为-1.0（初始值或未测得），则显示 "N/A"
        dist_str = f"{distance_cm:.2f} cm" if distance_cm > 0 else "N/A"
        size_str = f"{size_x_cm:.2f} cm" if size_x_cm > 0 else "N/A"

        # 2. 创建并发送距离指令
        # 格式: t0.txt="xx.xx cm"
        cmd_dist = f't0.txt="{dist_str}"'
        ser_sender.write(cmd_dist.encode("gb2312")) # 使用GB2312编码
        ser_sender.write(end_cmd) # 发送结束符

        # 3. 创建并发送尺寸指令
        # 格式: t1.txt="xx.xx cm"
        cmd_size = f't1.txt="{size_str}"'
        ser_sender.write(cmd_size.encode("gb2312")) # 使用GB2312编码
        ser_sender.write(end_cmd) # 发送结束符

    except serial.SerialException as e:
        print(f"串口屏发送错误: {e}")