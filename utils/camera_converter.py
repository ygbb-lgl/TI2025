import cv2
import numpy as np
# 示例：已知深度的情况下进行坐标转换
# converter = CameraConverter()
# converter.current_camera = 1

# 场景1: 已知像素坐标和深度，求世界坐标
# u, v = 320, 240
# depth_z = 1000  # 深度1000mm
# X, Y, Z = converter.pixel_to_world_coords(u, v, depth_z)
# print(f"像素({u}, {v}) + 深度{depth_z}mm -> 世界坐标({X:.2f}, {Y:.2f}, {Z:.2f})")

class CameraConverter:
    """
    摄像头坐标转换器
    支持多摄像头切换和像素坐标到相机坐标的转换
    """
    
    def __init__(self):
        """初始化摄像头配置"""
        self.CAMERA_CONFIGS = {
            '1_640': {
                "name": "摄像头1 (有畸变)",
                "mtx": np.array([
                    [494.85400649, 0, 319.85502813],
                    [0, 490.19647267, 230.51331758],
                    [0, 0, 1]
                ], dtype=np.float32),
                "dist": np.array([[-0.42830145, 0.23527948, 0.01128685, 0.00068342, -0.19925847]], dtype=np.float32),
                "has_distortion": True
            },
            '2_640': {
                "name": "摄像头2 (无畸变)",
                "mtx": np.array([
                     [640.38363912 ,  0        ,319.99999154],
                     [  0      ,   639.31966916 ,240.00000164],
                     [  0       ,    0          , 1        ]
                ], dtype=np.float32),
                "dist": np.array([[0, 0, 0, 0, 0]], dtype=np.float32),
                "has_distortion": False
            },
            '2_1080': {
                "name": "摄像头2_1080 (无畸变)",
                "mtx": np.array([
                    [1.91981113e+03, 0.00000000e+00 ,9.60000014e+02],
                    [0.00000000e+00 ,1.92059740e+03 ,5.40000000e+02],
                    [0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00]
                ], dtype=np.float32),
                "dist": np.array([[0,0,0,0,0]], dtype=np.float32),
                "has_distortion": False
            }
        }
        
        self._current_camera = '2_640'  # 默认使用摄像头2
    
    @property
    def current_camera(self):
        """获取当前摄像头编号"""
        return self._current_camera
    
    @current_camera.setter
    def current_camera(self, camera_id):
        """
        设置当前摄像头
        
        Args:
            camera_id (int): 摄像头编号 (1 或 2)
        
        Raises:
            ValueError: 如果摄像头编号无效
        """
        if camera_id not in self.CAMERA_CONFIGS:
            raise ValueError(f"无效的摄像头编号: {camera_id}. 支持的编号: {list(self.CAMERA_CONFIGS.keys())}")
        
        self._current_camera = camera_id
        print(f"已切换到: {self.CAMERA_CONFIGS[camera_id]['name']}")
    
    def get_camera_config(self, camera_id=None):
        """
        获取摄像头配置
        
        Args:
            camera_id (int, optional): 摄像头编号，为None时使用当前摄像头
        
        Returns:
            dict: 摄像头配置字典
        """
        if camera_id is None:
            camera_id = self._current_camera
        
        if camera_id not in self.CAMERA_CONFIGS:
            raise ValueError(f"无效的摄像头编号: {camera_id}")
        
        return self.CAMERA_CONFIGS[camera_id]
    
    def pixel_to_camera(self, u, v, camera_id=None):
        """
        将像素坐标转换为归一化相机坐标
        
        Args:
            u (float): 像素坐标u
            v (float): 像素坐标v
            camera_id (int, optional): 摄像头编号，为None时使用当前摄像头
        
        Returns:
            tuple: (x, y) 归一化相机坐标
        """
        config = self.get_camera_config(camera_id)
        mtx = config["mtx"]
        dist = config["dist"]
        has_distortion = config["has_distortion"]
        
        if has_distortion:
            # 有畸变的情况，使用OpenCV去畸变
            pts = np.array([[[u, v]]], dtype=np.float32)
            undistorted = cv2.undistortPoints(pts, mtx, dist)
            x, y = undistorted[0, 0]
            return float(x), float(y)
        else:
            # 无畸变的情况，直接用内参矩阵计算
            fx = mtx[0, 0]
            fy = mtx[1, 1]
            cx = mtx[0, 2]
            cy = mtx[1, 2]
            x = (u - cx) / fx
            y = (v - cy) / fy
            return float(x), float(y)
    
    def update_camera_params(self, camera_id, fx, fy, cx, cy, dist=None):
        """
        更新指定摄像头的内参
        
        Args:
            camera_id (int): 摄像头编号
            fx (float): 焦距fx
            fy (float): 焦距fy
            cx (float): 主点cx
            cy (float): 主点cy
            dist (array, optional): 畸变系数，为None时保持原有设置
        """
        if camera_id not in self.CAMERA_CONFIGS:
            raise ValueError(f"无效的摄像头编号: {camera_id}")
        
        self.CAMERA_CONFIGS[camera_id]["mtx"] = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        if dist is not None:
            self.CAMERA_CONFIGS[camera_id]["dist"] = np.array([dist], dtype=np.float32)
            self.CAMERA_CONFIGS[camera_id]["has_distortion"] = not np.allclose(dist, 0)
        
        print(f"摄像头{camera_id}内参已更新")
    
    def print_camera_info(self, camera_id=None):
        """
        打印摄像头配置信息
        
        Args:
            camera_id (int, optional): 摄像头编号，为None时使用当前摄像头
        """
        config = self.get_camera_config(camera_id)
        camera_id = camera_id or self._current_camera
        
        print(f"\n=== 摄像头{camera_id}配置信息 ===")
        print(f"名称: {config['name']}")
        print(f"内参矩阵:\n{config['mtx']}")
        if config['has_distortion']:
            print(f"畸变系数:\n{config['dist']}")
        else:
            print("无畸变")
    
    def get_available_cameras(self):
        """
        获取可用的摄像头列表
        
        Returns:
            list: 摄像头编号列表
        """
        return list(self.CAMERA_CONFIGS.keys())
    
    def pixel_to_world_coords(self, u, v, depth_z, camera_id=None):
        """
        将像素坐标转换为世界坐标 (需要深度信息Z)
        
        Args:
            u (float): 像素坐标u
            v (float): 像素坐标v
            depth_z (float): 深度值Z (单位: mm 或与内参一致的单位)
            camera_id (int, optional): 摄像头编号，为None时使用当前摄像头
        
        Returns:
            tuple: (X, Y, Z) 世界坐标
        """
        # 先获取归一化相机坐标 (x, y) = (X/Z, Y/Z)
        x_norm, y_norm = self.pixel_to_camera(u, v, camera_id)
        
        # 根据 X/Z = x_norm, Y/Z = y_norm 计算实际坐标
        X = x_norm * depth_z
        Y = y_norm * depth_z
        Z = depth_z
        
        return float(X), float(Y), float(Z)

    def world_to_pixel(self, X, Y, Z, camera_id=None):
        """
        将世界坐标转换为像素坐标 (逆向转换)
        
        Args:
            X, Y, Z (float): 世界坐标
            camera_id (int, optional): 摄像头编号
        
        Returns:
            tuple: (u, v) 像素坐标
        """
        config = self.get_camera_config(camera_id)
        mtx = config["mtx"]
        dist = config["dist"]
        has_distortion = config["has_distortion"]
        
        # 归一化相机坐标
        x_norm = X / Z
        y_norm = Y / Z
        
        if has_distortion:
            # 有畸变的情况，使用OpenCV投影
            object_points = np.array([[X, Y, Z]], dtype=np.float32)
            rvec = np.zeros(3, dtype=np.float32)  # 假设相机坐标系
            tvec = np.zeros(3, dtype=np.float32)
            
            image_points, _ = cv2.projectPoints(object_points, rvec, tvec, mtx, dist)
            u, v = image_points[0, 0]
            return float(u), float(v)
        else:
            # 无畸变的情况，直接用内参计算
            fx = mtx[0, 0]
            fy = mtx[1, 1]
            cx = mtx[0, 2]
            cy = mtx[1, 2]
            
            u = x_norm * fx + cx
            v = y_norm * fy + cy
            return float(u), float(v)