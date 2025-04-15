import open3d as o3d
import copy
from util import *

gt_key_path = "build/rectangle_center_keypoints.pcd"
pred_key_path = "build/largest_cluster_rect_keypoints.pcd"

gt_pcd = o3d.io.read_point_cloud(gt_key_path)
gt_points = np.array(gt_pcd.points)

gt_pcd.paint_uniform_color([0.0, 1.0, 0.0])

pred_pcd = o3d.io.read_point_cloud(pred_key_path)
pred_points = np.array(pred_pcd.points)
pred_pcd.paint_uniform_color([1.0, 0.0, 0.0])


# visualize_pcd(gt_key_path)
# visualize_pcd(pred_key_path)

o3d.visualization.draw_geometries([pred_pcd, gt_pcd])




init_trans = get_FPFH(gt_pcd, pred_pcd)

transfered_gt_pcd = copy.deepcopy(gt_pcd)
transfered_gt_pcd.transform(init_trans)

transfered_gt_pcd.paint_uniform_color([0.0, 0.0, 1.0])



o3d.visualization.draw_geometries([transfered_gt_pcd, pred_pcd, gt_pcd])


 

R_init = init_trans[:3, :3]
t_init = init_trans[:3, 3]

threshold = 25
max_iter = 30


best_t, best_R, correspondences = ICP_algorithm(gt_points, pred_points, t_init, R_init, threshold, max_iter)

# print("best_t is: ", best_t)
# print("best_R is: ", best_R)

final_trandform = np.eye(4)
final_trandform[:3, :3] = best_R
final_trandform[:3, 3]  = best_t

final_transfered_gt = copy.deepcopy(gt_pcd)
final_transfered_gt.transform(final_trandform)
final_transfered_gt.paint_uniform_color([0.0, 0.0, 1.0])

o3d.visualization.draw_geometries([final_transfered_gt, pred_pcd, gt_pcd])
 