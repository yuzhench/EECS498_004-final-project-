# depth2pointcloud.py 
input: depth image \
output: pointcloud (o3d) 

# keypointPCL.cpp
input: point cloud (dense PCD) \
output: key point cloud (sparce PCD ) 

# text2mask.py 
input: rgb image, prompt word. \
output: mask of the interested target. 



![rgb_image](./media/rgb.png "rgb_iamge")
![depth_image](./media/depth.png "depth_iamge")

![image_1](./media/1.png "first image")
![image_2](./media/2.png "second image")
![image_3](./media/3.png "third image")

### First Image
- **Green**: Ground truth  
- **Red**: Point cloud derived from converting the real-world depth image into a point cloud  

### Second Image
- **Green** and **Red**: Same meaning and positions as in the first image  
- **Blue**: An initial, relatively rough \(R\) and \(T\) result from FPFH feature matching and iterative alignment.  
  - This \(R\) and \(T\) only moves the ground truth to a position roughly close to the real-world captured point cloud.  

### Third Image
- **Green** and **Red**: Meanings and positions unchanged  
- **Blue**: The final result after ICP.  
  - Compared to the second image, it’s more accurate.  
  - There are still issues (e.g., the base is flipped), but that doesn’t matter much because we mainly care about the box’s orientation. As long as the plane’s normal vector is correct, it’s acceptable.
