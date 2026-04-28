import serial

def send_data(ser_hmi, control_id, attribute, value):
    """
    向串口屏发送通用更新指令。

    :param ser_hmi: 已打开的 serial.Serial 对象，用于与HMI（串口屏）通信。
    :param page: 控件所在的页面名称, 例如 "page0"。
    :param control_id: 控件的ID, 例如 "t0" (文本框), "n1" (数字框)。
    :param attribute: 要修改的控件属性, 例如 "txt" (文本), "val" (数值)。
    :param value: 要设置的值。如果是文本，则为字符串；如果是数值，则为数字。
    """

    # 如果数据为-1则不发送
    if value == -1 or value == -8.958:
        print(f"数据为-1，跳过发送 {control_id}.{attribute}")
        return

    if not (ser_hmi and ser_hmi.is_open):
        print("HMI串口未打开或无效，跳过发送。")
        return

    # 定义串口屏指令的结束符
    end_cmd = bytes.fromhex('ff ff ff')

    # 根据属性类型格式化值
    # 文本属性(.txt)的值需要用双引号括起来
    if attribute == 'txt':
        final_value = f'"{str(value)}"'
    # 数值属性(.val)的值直接是数字
    else:
        final_value = str(value)

    # 构建完整的指令字符串
    # 格式示例: page0.t0.txt="Hello" 或 page0.n0.val=123
    command_str = f'{control_id}.{attribute}={final_value}'

    try:
        # 打印调试信息，方便观察发送了什么
        # print(f"发送指令: {command_str}")
        
        # 发送指令
        ser_hmi.write(command_str.encode("utf-8"))
        ser_hmi.write(end_cmd)

    except serial.SerialException as e:
        print(f"HMI串口屏发送错误: {e}")
