#=== vẽ OBB
import numpy as np
import open3d
def get_horizontal_obb(pcd):
    pts = np.asarray(pcd.points)

    # 0. Cluster center
    center = pts.mean(axis=0)

    # 1. Work in local coordinates (centered)
    pts_local = pts - center

    # 2. PCA on XY plane (local)
    pts_xy = pts_local[:, :2]
    cov = np.cov(pts_xy.T)
    eigvals, eigvecs = np.linalg.eig(cov)

    # Principal direction in XY
    direction = eigvecs[:, np.argmax(eigvals)]
    yaw = np.arctan2(direction[1], direction[0])

    # 3. Rotation around Z only
    R = np.array([
        [ np.cos(yaw), -np.sin(yaw), 0],
        [ np.sin(yaw),  np.cos(yaw), 0],
        [ 0,            0,           1]
    ])

    # 4. Rotate local points
    pts_rot = (R.T @ pts_local.T).T  # shape (N,3)

    # 5. AABB in rotated frame
    min_bound = pts_rot.min(axis=0)
    max_bound = pts_rot.max(axis=0)
    extent = max_bound - min_bound
    center_rot = 0.5 * (min_bound + max_bound)

    # 6. Map box center back to world coordinates
    center_world = center + R @ center_rot

    # 7. Build horizontal OBB (upright)
    obb = open3d.geometry.OrientedBoundingBox(center_world, R, extent)
    obb.color = (0, 0, 1)
    return obb
