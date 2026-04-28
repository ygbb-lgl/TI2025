import numpy as np
#八个点坐标，矩形框外四个点，框内四个点，已知矩形框长宽和黑色边框线宽度
def normalize_points(points):
    """
    对点集进行归一化，使其均值为0，均方根为sqrt(2)
    :param points: Nx2数组
    :return: 归一化后的点、归一化矩阵T
    """
    mean = np.mean(points, axis=0)
    std = np.std(points)
    scale = np.sqrt(2) / std
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    points_norm = (T @ points_h.T).T
    return points_norm[:, :2], T

def compute_homography(world_points, image_points):
    """
    计算单应性矩阵H（带归一化DLT）
    :param world_points: 4x2的世界坐标点数组，每行为(X, Y)
    :param image_points: 4x2的像素坐标点数组，每行为(u, v)
    :return: 3x3 单应性矩阵 H
    """
    # 归一化
    world_points_norm, T1 = normalize_points(np.array(world_points))
    image_points_norm, T2 = normalize_points(np.array(image_points))

    A = []
    for i in range(len(world_points)):
        X, Y = world_points_norm[i]
        u, v = image_points_norm[i]
        A.append([X, Y, 1, 0, 0, 0, -u*X, -u*Y, -u])
        A.append([0, 0, 0, X, Y, 1, -v*X, -v*Y, -v])
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    h = V[-1, :]
    H_norm = h.reshape(3, 3)
    # 反归一化
    H = np.linalg.inv(T2) @ H_norm @ T1
    H = H / H[2, 2]
    return H

def invert_homography(image_points, H):
    """
    已知像素坐标和单应性矩阵H，反解世界坐标
    :param image_points: 4x2的像素坐标点数组，每行为(u, v)
    :param H: 3x3 单应性矩阵
    :return: 4x2的世界坐标点数组，每行为(X, Y)
    """
    H_inv = np.linalg.inv(H)
    world_points = []
    for u, v in image_points:
        uv1 = np.array([u, v, 1.0])
        XY1 = H_inv @ uv1
        XY1 = XY1 / XY1[2]
        world_points.append([XY1[0], XY1[1]])
    return np.array(world_points)