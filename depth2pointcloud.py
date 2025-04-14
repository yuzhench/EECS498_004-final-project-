import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import matplotlib.pyplot as plt
# from google.colab import drive
from tqdm import tqdm
import open3d as o3d
from PIL import Image
from util import *

def generate_point_clouds(rgb_image, depth_image, mask, camera_data):

    print("rgb_image is: ",rgb_image.shape)
    print("the mask shape is: ", mask.shape)
    print("the depth image shape is: ", depth_image.shape)

    

    #let the input rgb_image, depth_image tobe [H, W, 3]
    H = rgb_image.shape[0]
    W = rgb_image.shape[1]

    #initialization 
    point_cloud = []
    rgb = []

    cx = camera_data["cx"]
    cy = camera_data["cy"]
    fx = camera_data["fx"]
    fy = camera_data["fy"]

    # camera_center = camera_data["camera_center"].reshape(-1) 
   
    
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,   0, 1 ]
    ])

    K_inv = np.linalg.inv(K)

    #get the valid rgb_imge from th original rgb image 
    for row_i in range(H):
        for col_i in range(W):
            if mask[row_i][col_i] != 0:
                
                Z = (depth_image[row_i,col_i].astype(np.float32))
                pixel_hom = np.array([col_i,row_i, 1], dtype=np.float32)
                
                direction = K_inv @ pixel_hom
                point_3D = direction * Z
                X = point_3D[0]
                Y = point_3D[1]
                Z = point_3D[2]
              
                point_shifted  = [X,Y,Z] #- camera_center

                point_cloud.append(point_shifted)

                rgb.append(rgb_image[row_i,col_i])

    point_cloud = np.array(point_cloud, dtype=np.float32) 
    rgb = np.array(rgb, dtype=np.float32) 


    return point_cloud, rgb



 
def visualizing_point_clouds(point_cloud,rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    # pcd.colors = o3d.utility.Vector3dVector(rgb)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1,  
        origin=[0, 0, 0]
    )

    o3d.visualization.draw_geometries([pcd,axis])
    # o3d.visualization.draw_plotly([pcd])

    return None




def main():

 
    rgb_image = Image.open("rgb.png")
    rgb_image = np.array(rgb_image)


    depth_image = Image.open("depth.png")
    depth_image = np.array(depth_image)


    # mask = point_cloud_data["mask"]
    # mask = np.ones((rgb_image.shape[0], rgb_image.shape[1]))
    
    mask = get_mask()
    print(mask.shape)
    plt.imshow(mask, cmap="gray")
    plt.show()

    

    image_H = 720
    image_W = 1280
    #point: 0.4224, -0.0300
    camera_data = {
        "cx": 641.7578,
        "cy": 365.6518,
        "fx": 606.1124,
        "fy": 605.8821,
    }
    print(rgb_image.shape)
 

    point_cloud, rgb = generate_point_clouds(rgb_image, depth_image, mask, camera_data)

    print("the shape of the point_cloud is: ", point_cloud.shape)
 

    visualizing_point_clouds(point_cloud, rgb)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)


    """check the range"""
    print("Min bound:", pcd.get_min_bound())
    print("Max bound:", pcd.get_max_bound())
 
    
    labels = np.array(pcd.cluster_dbscan(eps=5, min_points=20, print_progress=True))
    unique_labels = np.unique(labels)


    print("lable shape is: ", labels.shape)
    print("unique_labels shape is: ", unique_labels)


    valid_mask = (labels >= 0)
    if not np.any(valid_mask):
        print("all the points are noise")
    else:
        valid_labels = labels[valid_mask]
        largest_label = np.argmax(np.bincount(valid_labels))
        largest_indices = np.where(labels == largest_label)[0]
        largest_cluster_pcd = pcd.select_by_index(largest_indices)
        largest_cluster_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        
        print(f"the biggest lable is  {largest_label}, total {len(largest_indices)} points。")
        #visualization 
        o3d.visualization.draw_geometries([largest_cluster_pcd])
     



if __name__ == "__main__":
    main()