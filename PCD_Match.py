import open3d as o3d
import copy
import numpy as np
from util import *


def normalize_scale(pcd):
    pts = np.asarray(pcd.points)
    center = np.mean(pts, axis=0)
    pts -= center
    scale = np.linalg.norm(pts, axis=1).max()
    pts /= scale
    pts_normalized = pts + center

    pcd.points = o3d.utility.Vector3dVector(pts_normalized)
    return pcd

gt_key_path = "build/rectangle_center_keypoints.pcd"
pred_key_path = "build/example_iss_rectangle_keypoints00380.pcd"

# gt_pcd = o3d.io.read_point_cloud(gt_key_path)
# gt_points = np.array(gt_pcd.points)

# gt_pcd.paint_uniform_color([0.0, 1.0, 0.0])


# pred_pcd = o3d.io.read_point_cloud(pred_key_path)
# pred_points = np.array(pred_pcd.points)
# pred_pcd.paint_uniform_color([1.0, 0.0, 0.0])


# # visualize_pcd(gt_key_path)
# # visualize_pcd(pred_key_path)

# o3d.visualization.draw_geometries([pred_pcd, gt_pcd])




# init_trans = get_FPFH(gt_pcd, pred_pcd)

# transfered_gt_pcd = copy.deepcopy(gt_pcd)
# transfered_gt_pcd.transform(init_trans)

# transfered_gt_pcd.paint_uniform_color([0.0, 0.0, 1.0])



# o3d.visualization.draw_geometries([transfered_gt_pcd, pred_pcd, gt_pcd])


 

# R_init = init_trans[:3, :3]
# t_init = init_trans[:3, 3]

# threshold = 25
# max_iter = 30


# best_t, best_R, correspondences = ICP_algorithm(gt_points, pred_points, t_init, R_init, threshold, max_iter)

# # print("best_t is: ", best_t)
# # print("best_R is: ", best_R)

# final_trandform = np.eye(4)
# final_trandform[:3, :3] = best_R
# final_trandform[:3, 3]  = best_t

# final_transfered_gt = copy.deepcopy(gt_pcd)
# final_transfered_gt.transform(final_trandform)
# final_transfered_gt.paint_uniform_color([0.0, 0.0, 1.0])

# o3d.visualization.draw_geometries([final_transfered_gt, pred_pcd, gt_pcd])
 

gt_pcd = o3d.io.read_point_cloud(gt_key_path)
pred_pcd = o3d.io.read_point_cloud(pred_key_path)

# Normalize scale (important step)
gt_pcd = normalize_scale(gt_pcd)
pred_pcd = normalize_scale(pred_pcd)

# Paint colors
gt_pcd.paint_uniform_color([0.0, 1.0, 0.0])   # Green
pred_pcd.paint_uniform_color([1.0, 0.0, 0.0]) # Red

# Visualize original (normalized) point clouds
o3d.visualization.draw_geometries([pred_pcd, gt_pcd])

# Run FPFH-based initial alignment
init_trans = get_FPFH(gt_pcd, pred_pcd)

# Apply initial transformation to GT
transfered_gt_pcd = copy.deepcopy(gt_pcd)
transfered_gt_pcd.transform(init_trans)
transfered_gt_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # Blue

# Visualize initial alignment
o3d.visualization.draw_geometries([transfered_gt_pcd, pred_pcd, gt_pcd])

# Extract rotation and translation
R_init = init_trans[:3, :3]
t_init = init_trans[:3, 3]

# Run custom ICP refinement
threshold = 25
max_iter = 30
gt_points = np.array(gt_pcd.points)
pred_points = np.array(pred_pcd.points)

best_t, best_R, correspondences = ICP_algorithm(gt_points, pred_points, t_init, R_init, threshold, max_iter)

# Construct final transformation matrix
final_trandform = np.eye(4)
final_trandform[:3, :3] = best_R
final_trandform[:3, 3] = best_t

# Apply final transformation
final_transfered_gt = copy.deepcopy(gt_pcd)
final_transfered_gt.transform(final_trandform)
final_transfered_gt.paint_uniform_color([0.0, 0.0, 1.0])  # Blue again



# def get_average_normal_arrow (pcd, length=1,radius = 0.05, color = [0, 0, 1]):
#     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=180))
#     pcd.normalize_normals()
#     normals = np.asarray(pcd.normals)
#     avg_normal = np.mean(normals, axis=0)
#     avg_normal /= np.linalg.norm(avg_normal)
#     print(f"Average normal vector: {avg_normal}")
#     # Get centroid and direction
#     origin = np.mean(np.asarray(pcd.points), axis=0)
#     direction = avg_normal

#     # Create arrow mesh
#     arrow = o3d.geometry.TriangleMesh.create_arrow(
#         cylinder_radius=radius,
#         cone_radius=radius * 2,
#         cylinder_height=length * 0.8,
#         cone_height=length * 0.2
#     )
#     arrow.paint_uniform_color(color)

#     # Align with direction
#     rot_matrix = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, 0])  # default
#     z_axis = np.array([0, 0, 1])
#     axis = np.cross(z_axis, direction)
#     angle = np.arccos(np.dot(z_axis, direction))
#     if np.linalg.norm(axis) > 1e-6:
#         axis /= np.linalg.norm(axis)
#         rot_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
#     arrow.rotate(rot_matrix, center=(0, 0, 0))
#     arrow.translate(origin)
    
    
    # return arrow
    # Final visualization

def get_average_normal_vector(pcd):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=180))
    pcd.normalize_normals()
    normals = np.asarray(pcd.normals)
    avg_normal = np.mean(normals, axis=0)
    avg_normal /= np.linalg.norm(avg_normal)
    return avg_normal

# gt_arrow, pred_arrow, final_arrow = get_average_normal_arrow(gt_pcd, color = [0,1,0]), get_average_normal_arrow(pred_pcd, color = [0,1,0]), get_average_normal_arrow(final_transfered_gt, color = [0,1,0])
gt_normal = get_average_normal_vector(gt_pcd)
final_normal = get_average_normal_vector(final_transfered_gt)

print("GT normal vector:", gt_normal)
print("Final transformed GT normal vector:", final_normal)

# o3d.visualization.draw_geometries(
#     [final_transfered_gt, pred_pcd, gt_pcd, gt_arrow, pred_arrow, final_arrow],
#     )