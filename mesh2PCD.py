import open3d as o3d
 
import numpy as np

def mesh_to_PCD(mesh_name): 
    # 1. 读取网格文件
    mesh = o3d.io.read_triangle_mesh(f"{mesh_name}.obj")
    mesh.compute_vertex_normals()  # 可选：计算法线，用于采样算法

    # 2. 将网格采样成点云
    #   方法1：Poisson Disk 采样
    pcd_poisson = mesh.sample_points_poisson_disk(number_of_points=10000)

    #   方法2：均匀采样
    pcd_uniform = mesh.sample_points_uniformly(number_of_points=10000)

    # 3. 查看采样结果
    print("Poisson sampled point cloud has", len(pcd_poisson.points), "points")
    print("Uniform sampled point cloud has", len(pcd_uniform.points), "points")
    pcd_file = f"{mesh_name}.pcd"
    o3d.io.write_point_cloud(pcd_file, pcd_poisson)
    print(f"Saved Poisson point cloud to: {pcd_file}")

    # 4. 可视化
    o3d.visualization.draw_geometries([pcd_poisson])


 
def plot_normal_vector(PCD_file):
    # 1. 读取点云
    pcd = o3d.io.read_point_cloud(PCD_file)  # 或者 .ply / .xyz 等

    # 2. 估计法线
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )

    # 3. 可视化
    # （a）直接可视化点云和法线
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    # （b）若想画出点云的主方向，可先做 PCA 或用 open3d 的 get_oriented_bounding_box()
    obb = pcd.get_oriented_bounding_box()
    obb.color = (1, 0, 0)  # 先给个红色，方便看

    # 将 OBB 也加入可视化
    o3d.visualization.draw_geometries([pcd, obb])


def rotation_matrix_from_vectors(vec1, vec2):
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)

    # 若两向量几乎相反
    if np.isclose(c, -1.0):
        # 选个任意正交向量做旋转轴
        # 这里简单返回一个负单位阵
        return -np.eye(3)

    # 若两向量几乎相同
    if np.isclose(c, 1.0):
        # 不需要旋转
        return np.eye(3)

    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return R


if __name__ == "__main__":
    # mesh_file_name = "rectangle.obj"
    # PCD_file_name = "rectangle.pcd"
    mesh_to_PCD("rectangle_center")
    # plot_normal_vector("trangle.pcd")
 

 

    # # 1. 读取点云
    # pcd = o3d.io.read_point_cloud(PCD_file_name)

    # # 2. 估计法线（若还没估过）
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    # # 3. 计算整体中心
    # points = np.asarray(pcd.points)
    # print("points shape: ", points)
    # center = points.mean(axis=0)

    # print("the center is: ", center)

 

    # # 4. 得到“整体法向量”
    # #    - 这里用所有点的法线均值做一个近似，也可以用PCA等方式
    # normals = np.asarray(pcd.normals)
    # mean_normal = normals.mean(axis=0)
    # mean_normal /= np.linalg.norm(mean_normal)  # 归一化


    # print("the mean normal vector is: ", mean_normal)
 

    # # 5. 创建一个箭头，用来表示法向量
    # #    - create_arrow 参数可自定义，例如圆柱/圆锥高度、半径等等
    # arrow_length = 300  # 箭头长度可根据物体尺寸适当调整
    # arrow = o3d.geometry.TriangleMesh.create_arrow(
    #     cone_height=0.06,
    #     cone_radius=0.03,
    #     cylinder_height=0.14,
    #     cylinder_radius=0.01
    # )

    # arrow.paint_uniform_color([0.0, 1.0, 0.0])

    # # 5.1 调整箭头整体尺寸（让它乘以一个系数来拉长或缩短）
    # # arrow.scale(arrow_length, center=False)
    # arrow.scale(arrow_length, center=(center/100))


    # # 5.2 默认情况下，create_arrow()生成的箭头指向 +Z 方向
    # #     现在要把+Z轴旋转到 mean_normal 的方向

    # # z_axis = np.array([0, 0, 1], dtype=float)
    # # R = rotation_matrix_from_vectors(z_axis, mean_normal)
    # # arrow.rotate(R, center=center)

    # # 5.3 把箭头平移到点云中心
    # # arrow.translate(center)
 

    # # 6. 可视化
    # #    - 可以选择只显示部分法线箭头，避免太乱
    # # pcd_for_viz = pcd.copy()
    # # pcd_for_viz.paint_uniform_color([0.8, 0.8, 0.8])  # 给点云一个中性色


    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=1.0,
    #     origin=[0.0, 0.0, 0.0]
    # )
        

    # o3d.visualization.draw_geometries(
    #     [arrow, pcd, axis],
    #     point_show_normal=False  # 可以关掉每个点的小法线显示
    # )






