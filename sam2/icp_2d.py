import numpy as np
import cv2
from scipy.spatial import KDTree
from scipy.linalg import svd

def extract_points_from_mask(mask, edges_only=True):
    """Extracts edge points from a binary mask."""
    if edges_only:
        edges = cv2.Canny(mask.astype(np.uint8) * 255, 100, 200)
        points = np.column_stack(np.where(edges > 0))
    else:
        points = np.column_stack(np.where(mask > 0))
    # print(len(points))
    return points

def best_fit_transform(A, B):
    """Computes the best-fit affine transformation (rotation + translation)."""
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    
    H = A_centered.T @ B_centered
    U, _, Vt = svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    t = centroid_B - R @ centroid_A
    
    T = np.eye(4)
    T[:2, :2] = R
    T[:2, 3] = t
    return T

def icp(source_mask, target_mask, max_iterations=50, tolerance=1e-5, edges_only=True):
    """Performs ICP to align source_mask to target_mask and returns a 4x4 transformation matrix."""
    source_points = extract_points_from_mask(source_mask, edges_only)
    target_points = extract_points_from_mask(target_mask, edges_only)
    
    prev_error = float('inf')
    T = np.eye(4)
    
    for i in range(max_iterations):
        tree = KDTree(target_points)
        distances, indices = tree.query(source_points)
        matched_target_points = target_points[indices]
        
        T_iter = best_fit_transform(source_points, matched_target_points)
        
        source_points = (T_iter[:2, :2] @ source_points.T).T + T_iter[:2, 3]
        T = T_iter @ T
        
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    
    return T
