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
import glob

def generate_point_clouds(rgb_image, depth_image, mask, camera_data, MAX_DEPTH = 150):

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
                if Z > MAX_DEPTH:
                    continue
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


def process_frame(rgb_path, depth_path, mask_path, camera_data, visualize=False, save_output=True):
    """Process a single frame and generate point cloud"""
    rgb_image = Image.open(rgb_path)
    rgb_image = np.array(rgb_image)
    
    depth_image = Image.open(depth_path)
    depth_image = np.array(depth_image)
    
    mask = np.load(mask_path)
    
    point_cloud, rgb = generate_point_clouds(rgb_image, depth_image, mask, camera_data)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)  # Normalize RGB values
    
    labels = np.array(pcd.cluster_dbscan(eps=5, min_points=20, print_progress=False))
    
    valid_mask = (labels >= 0)
    if not np.any(valid_mask):
        print(f"[Warning] All points classified as noise for {os.path.basename(rgb_path)}")
        return None
    else:
        valid_labels = labels[valid_mask]
        largest_label = np.argmax(np.bincount(valid_labels))
        largest_indices = np.where(labels == largest_label)[0]
        largest_cluster_pcd = pcd.select_by_index(largest_indices)
        largest_cluster_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        
        if save_output:
            frame_num = os.path.basename(rgb_path).split('.')[0]
            output_path = f"output_pointclouds/cloud_{frame_num}.ply"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            o3d.io.write_point_cloud(output_path, largest_cluster_pcd)
        
        if visualize:
            o3d.visualization.draw_geometries([largest_cluster_pcd])
        
        return largest_cluster_pcd

def main():
    rgb_dir = "/home/anranli/Documents/DeepL/Final/Final Project Demo/frames"
    depth_dir = "/home/anranli/Documents/DeepL/Final/Final Project Demo/depth_maps"
    mask_dir = "/home/anranli/Documents/DeepL/Final/Final Project Demo/binary_masks"
    output_dir = "/home/anranli/Documents/DeepL/Final/Final Project Demo/output_pointclouds"
    
    os.makedirs(output_dir, exist_ok=True)
    
    camera_data = {
        "cx": 641.7578,
        "cy": 365.6518,
        "fx": 606.1124,
        "fy": 605.8821,
    }
    
    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
    
    print(f"Found {len(rgb_paths)} frames to process.")
    
    for rgb_path in tqdm(rgb_paths):
        frame_basename = os.path.basename(rgb_path)
        frame_num = frame_basename.split('.')[0]
        
        depth_path = os.path.join(depth_dir, f"depth_{frame_num}.png")
        mask_path = os.path.join(mask_dir, f"mask_{frame_num}.npy")
        
        if not os.path.exists(depth_path):
            print(f"[Warning] Depth map missing for frame {frame_num}")
            continue
        
        if not os.path.exists(mask_path):
            print(f"[Warning] Mask missing for frame {frame_num}")
            continue
        
        output_path = os.path.join(output_dir, f"cloud_{frame_num}.ply")
        
        if os.path.exists(output_path):
            print(f"[Info] Skipping frame {frame_num} - output already exists")
            continue
        
        try:
            process_frame(rgb_path, depth_path, mask_path, camera_data, 
                          visualize=False, save_output=True)
        except Exception as e:
            print(f"[Error] Failed to process frame {frame_num}: {e}")

def single_frame_test():
    """Test with a single frame for debugging"""
    rgb_path = "/home/anranli/Documents/DeepL/Final/Final Project Demo/frames/01445.jpg"
    depth_path = "/home/anranli/Documents/DeepL/Final/Final Project Demo/depth_maps/depth_1445.png"
    mask_path = "/home/anranli/Documents/DeepL/Final/Final Project Demo/binary_masks/mask_01445.npy"
    
    camera_data = {
        "cx": 641.7578,
        "cy": 365.6518,
        "fx": 606.1124,
        "fy": 605.8821,
    }
    
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
        print("Mask shape:", mask.shape)
        plt.figure(figsize=(10, 6))
        plt.imshow(mask, cmap="gray")
        plt.title("Loaded Mask")
        plt.colorbar()
        plt.show()
    else:
        print(f"Warning: Mask file not found at {mask_path}")
        rgb_image = np.array(Image.open(rgb_path))
        mask = np.ones((rgb_image.shape[0], rgb_image.shape[1]), dtype=bool)
        print("Created dummy mask with shape:", mask.shape)
    
    rgb_image = Image.open(rgb_path)
    rgb_image = np.array(rgb_image)
    print("RGB image shape:", rgb_image.shape)
    
    depth_image = Image.open(depth_path)
    depth_image = np.array(depth_image)
    print("Depth image shape:", depth_image.shape)
    
    point_cloud, rgb = generate_point_clouds(rgb_image, depth_image, mask, camera_data)
    print("Point cloud shape:", point_cloud.shape)
    print("RGB values shape:", rgb.shape)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    output_dir = "/home/anranli/Documents/DeepL/Final/Final Project Demo/output_pointclouds"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cloud_01163.pcd")
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Point cloud saved to {output_path}")
    
    visualizing_point_clouds(point_cloud, rgb)

if __name__ == "__main__":
    # Choose which mode to run
    # main()          
    single_frame_test()