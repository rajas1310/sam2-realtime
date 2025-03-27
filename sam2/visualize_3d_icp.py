import cv2
import open3d as o3d
import numpy as np

import icp_3d

from rajas_3d_sam2_video_inference import get_pcd, get_3Dmask_coords

depth_array = np.load("../../videos/new_recording_0324/record_gray_cylinder/raw_depth_frames.npy")
extrinsic = np.loadtxt("../../videos/new_recording_0324/extrinsic_params.txt")

FRAME_DIR = "../../videos/frames/"

frame1 = cv2.imread(f"{FRAME_DIR}/1_00418.jpg", 0)
frame2 = cv2.imread(f"{FRAME_DIR}/1_00428.jpg", 0)
dep_frame1 = depth_array[418-1]
dep_frame2 = depth_array[428-1]

ret, frame1 = cv2.threshold(frame1, 127, 255, cv2.THRESH_BINARY)
ret, frame2 = cv2.threshold(frame2, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("frame1", frame1)
cv2.imshow("frame2", frame2)

pcd_frame1 = get_pcd(frame1, dep_frame1, frame1, extrinsic_params=extrinsic)
pcd_frame2 = get_pcd(frame2, dep_frame2, frame2, extrinsic_params=extrinsic)

mask3d_f1 = get_3Dmask_coords(pcd_frame1)
mask3d_f2 = get_3Dmask_coords(pcd_frame2)

M = icp_3d.icp_3d(source_mask=mask3d_f1, target_mask=mask3d_f2, mask_coords=True)

def apply_transformation(points, T):
    """Applies a 3x3 affine transformation matrix to 2D points."""
    # Convert 2D points to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    homogeneous_points = np.hstack((points, ones))
    # print(points.shape)
    # print(homogeneous_points.shape)
    # print(homogeneous_points)
    
    # Apply transformation
    transformed_points = (T @ homogeneous_points.T).T
    
    # Convert back to 2D
    return transformed_points[:, :2]

def visualize_alignment(source_mask, target_mask, T):
    """Applies transformation and visualizes alignment."""
    source_points = icp_2d.extract_points_from_mask(source_mask)
    target_points = icp_2d.extract_points_from_mask(target_mask)
    
    # Transform source points
    transformed_points = apply_transformation(source_points, T)

    # Plot original source, transformed source, and target points
    plt.figure(figsize=(6, 8))
    plt.scatter(target_points[:, 1], target_points[:, 0], c='g', label="Target Edges", s=5)
    plt.scatter(source_points[:, 1], source_points[:, 0], c='r', label="Original Source", s=5)
    plt.scatter(transformed_points[:, 1], transformed_points[:, 0], c='b', label="Transformed Source", s=5, alpha=0.6)
    
    plt.legend()
    plt.gca().invert_yaxis()
    plt.title("ICP Alignment Verification")
    plt.show()

# Example usage
visualize_alignment(frame1, frame2, M)

cv2.waitKey(0)
cv2.destroyAllWindows()



