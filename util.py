from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os 
import torch 
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import open3d as o3d

"""helper function"""
np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()



def get_mask(): 
    """model selection"""
    sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    """device selection"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")


    """create the model"""
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    

    """load in the image"""
    image = Image.open('/home/yuzhench/Desktop/Course/ROB498-004/Project/Final_project/rgb.png')
    image = np.array(image.convert("RGB"))

    """show the image""" # for testing
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('on')


    predictor.set_image(image)

    input_point = np.array([[700, 400]])
    input_label = np.array([1])

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()


    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    print(masks.shape)

    output_mask = masks[0]
    return output_mask

    # show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)



def array2pcd(point_cloud_array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_array)
    return pcd


def visualize_pcd_path(file_path_list):
    
    pcd_list = []
    for file_path in file_path_list:
        #read the point cloud
        pcd = o3d.io.read_point_cloud(file_path)

        #print pcd info     
        print(f"Point cloud has {len(pcd.points)} points.")

        pcd_list.append(pcd)

    # visualization
    o3d.visualization.draw_geometries(pcd_list)



def estimate_correspondences(X, Y, t, R, threshold):
    """
    Estimate Correspondences between two point clouds.

    This function takes two point clouds, X and Y, along with an initial guess of
    translation 't' and rotation 'R', and a threshold value for the maximum distance
    between two points to consider them as correspondences.

    Parameters:
    X (numpy.ndarray): The first point cloud represented as an N x 3 numpy array,
                       where N is the number of points.
    Y (numpy.ndarray): The second point cloud represented as an M x 3 numpy array,
                       where M is the number of points.
    t (numpy.ndarray): The initial guess for translation, a 1 x 3 numpy array.
    R (numpy.ndarray): The initial guess for rotation, a 3 x 3 numpy array.
    threshold (float): The maximum distance between two points to consider them as
                       correspondences.

    Returns:
    correspondences (numpy.ndarray): A numpy array of estimated point correspondences, where each
                            correspondence is [x, y], where 'x' is the index of point
                            from point cloud X, and 'y' is is the index of a point from point cloud Y.
    """
    # correspondences = None
    #########################################
    #############YOUR CODE HERE##############
    #########################################

    #pointcloud Y should be the stable pointcloud 
    #pointcloud X should be transformed pointcloud 

    X_transformed = (R @ X.T).T + t

    correspondences_list = []
    for index, x in enumerate(X_transformed):
        
        #calculate the distance 
        diff = Y - x   #shape: (N,3)
        dists = np.linalg.norm(diff, axis=1) #(N,1)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        #check with the threshold 
        # print("min_dist is: ", min_dist)
        if min_dist <= threshold:
             
            correspondences_list.append([index, min_idx])
    
            
    correspondences = np.array(correspondences_list)
    ##########################################
    return correspondences


def compute_rigid_transformation(X,Y,correspondences):
    """
    Estimate the optimal rigid transformation between two point clouds.

    Given two point clouds X and Y, along with a list of estimated point correspondences,
    this function calculates the optimal rotation and translation that best aligns
    point cloud X with point cloud Y.

    Parameters:
    X (numpy.ndarray): The first point cloud represented as an N x 3 numpy array,
                       where N is the number of points.
    Y (numpy.ndarray): The second point cloud represented as an M x 3 numpy array,
                       where M is the number of points.
    correspondences (numpy.ndarray): A numpy array of estimated point correspondences, where each
                            correspondence is [x, y], where 'x' is the index of point
                            from point cloud X, and 'y' is is the index of a point from point cloud Y.

    Returns:
    rotation (numpy.ndarray): The estimated rotation matrix (3x3) that best aligns
                             point cloud X with point cloud Y.
    translation (numpy.ndarray): The estimated translation vector (1x3) that best
                                aligns point cloud X with point cloud Y.
    """
    # rotation,transformation = None
    #########################################
    #############YOUR CODE HERE##############
    #########################################

    #get the X Y correspondance 
    indices_X = correspondences[:, 0]
    indices_Y = correspondences[:, 1]

    X_points_set = X[indices_X]  
    Y_points_set = Y[indices_Y]  

    #calculate the centroid 
    centroid_X = np.mean(X_points_set, axis=0)
    centroid_Y = np.mean(Y_points_set, axis=0)
 

    X_points_set_centroid = X_points_set - centroid_X
    Y_points_set_centroid = Y_points_set - centroid_Y

    #create the matrix H 
    k = X_points_set_centroid.shape[0]
    H = (1.0 / k) * (X_points_set_centroid.T @ Y_points_set_centroid)

    #SVD find the best R T
    U, S, Vt = np.linalg.svd(H)

    rotation = Vt.T @ U.T

    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T

    transformation = centroid_Y - rotation @ centroid_X
    transformation = transformation.reshape(1, 3)
    ##########################################
    return rotation, transformation

def ICP_algorithm(X , Y , t, R, threshold, max_iter):
    #########################################
    #############YOUR CODE HERE##############
    #########################################
     
    for i in range(max_iter):
        correspondences = estimate_correspondences(X, Y, t, R, threshold)

        if correspondences.size == 0:
            print("No correspondences found in iteration", i)
            break

        print("the correspondences shape is: ", correspondences.shape)
        Curr_R, Curr_t = compute_rigid_transformation(X, Y, correspondences)
        # R = Curr_R @ R
        # t = (Curr_R @ t.T).T + Curr_t
        R = Curr_R
        t = Curr_t

    ##########################################
    return t,R,correspondences



def compute_cloud_resolution(pcd: o3d.geometry.PointCloud) -> float:
    if len(pcd.points) == 0:
        return 0.0
    
    #create the k-d tree
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    #init the resolution and point number 
    resolution = 0.0
    number_of_pairs = 0
    
    for i, point in enumerate(pcd.points):
        # find the current point + nearest neighbor
        [k, idx, dist_sq] = pcd_tree.search_knn_vector_3d(point, 2)
        
        #check if it find the cloest neighbor 
        if k == 2:
            resolution += np.sqrt(dist_sq[1])
            number_of_pairs += 1
    
    if number_of_pairs > 0:
        resolution /= number_of_pairs
    
    return resolution



def get_FPFH(keypts_gt, keypts_pred):

    keypts_gt_resolution = compute_cloud_resolution(keypts_gt)
    keypts_pred_resolution = compute_cloud_resolution(keypts_pred)

    print("keypts_gt_resolution is: ", keypts_gt_resolution)

 

    #calculate the normal vectr for each point 
    keypts_gt.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius= 3 * keypts_gt_resolution, max_nn=20)
    )
    keypts_pred.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius= 3 * keypts_pred_resolution, max_nn=20)
    )

  
    gt_radius_feature = 10 * keypts_gt_resolution
    pred_radius_feature = 10 * keypts_pred_resolution

    fpfh_gt = o3d.pipelines.registration.compute_fpfh_feature(
        keypts_gt,
        o3d.geometry.KDTreeSearchParamHybrid(radius=gt_radius_feature, max_nn=50)
    )
    fpfh_pred = o3d.pipelines.registration.compute_fpfh_feature(
        keypts_pred,
        o3d.geometry.KDTreeSearchParamHybrid(radius=pred_radius_feature, max_nn=50)
    )

    fpfh_gt_np = np.asarray(fpfh_gt.data)  
    print("fpfh_gt_np shape is:", fpfh_gt_np.shape)

    fpfh_pred_np = np.asarray(fpfh_pred.data)  
    print("fpfh_pred_np shape is:", fpfh_pred_np.shape)

    # ========== 4) RANSAC 全局配准，得到初始变换 ==========
    distance_threshold = 5  # 根据你的点云尺度调整
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        keypts_gt, keypts_pred, # 让 gt 做 source, pred 做 target --> target will not move 
        fpfh_gt, fpfh_pred,
        False,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,  # RANSAC中一次随机抽取多少对匹配来估计变换
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    print ("result_ransac.fitness is: ", result_ransac.fitness)
    init_trans = result_ransac.transformation
    print("Initial transformation from RANSAC:\n", init_trans)
    return init_trans

    # # ========== 5) 用 ICP 做局部精细化 ==========
    # #   - 这里为了精细配准，最好用完整的点云 (pcd_gt, pcd_pred) 而不是关键点云
    # icp_threshold = 0.2  # ICP距离阈值
    # result_icp = o3d.pipelines.registration.registration_icp(
    #     pcd_pred, pcd_gt, icp_threshold,
    #     init_trans,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint()
    # )

    # refined_trans = result_icp.transformation
    # print("Refined transformation via ICP:\n", refined_trans)

    # # ========== 6) 观察配准结果 ==========
    # pcd_pred_transformed = pcd_pred.transform(refined_trans)

    # # 可视化，看看对齐效果
    # o3d.visualization.draw_geometries([pcd_gt, pcd_pred_transformed])
