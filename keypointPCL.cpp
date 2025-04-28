#include <iostream>

// PCL header files
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <thread>   // C++11 thread library, used for sleep


double computeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    double resolution = 0.0;
    int number_of_pairs = 0;

    // Create a k-d tree for nearest neighbor search
    pcl::search::KdTree<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    // Traverse all points and find their nearest neighbor distance
    for (size_t i = 0; i < cloud->size(); ++i)
    {
        // Containers needed for nearest neighbor search
        std::vector<int> indices(2);
        std::vector<float> squaredDistances(2);

        // Find the nearest 2 neighbors (the point itself + its nearest neighbor)
        if (kdtree.nearestKSearch(cloud->at(i), 2, indices, squaredDistances) == 2)
        {
            // The first distance is 0, the second is the distance to the nearest neighbor
            resolution += std::sqrt(squaredDistances[1]);
            ++number_of_pairs;
        }
    }

    // Take the average value
    if (number_of_pairs != 0)
        resolution /= number_of_pairs;

    return resolution;
}

int main(int argc, char** argv)
{
    // If you want to pass parameters via the command line, you can change it to read from argv[1], etc.
    std::string input_pcd = "/home/anranli/Documents/DeepL/Final/Final Project Demo/output_pointclouds/cloud_00380.pcd";  
    std::string output_pcd = "example_iss_rectangle_keypoints00380.pcd";

    // 1. Define point cloud type and create pointers
    typedef pcl::PointXYZ PointT;
    pcl::PointCloud<PointT>::Ptr cloud_in(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr keypoints_out(new pcl::PointCloud<PointT>());

    // 2. Read the PCD file
    if (pcl::io::loadPCDFile<PointT>(input_pcd, *cloud_in) == -1)
    {
        PCL_ERROR("Unable to read file %s.\n", input_pcd.c_str());
        return -1;
    }

    std::cout << "Point cloud loaded successfully: " << input_pcd 
              << ", number of points: " << cloud_in->size() << std::endl;

    double calculated_reso = computeCloudResolution(cloud_in);
    std::cout << "calculated_reso is: " << calculated_reso << std::endl;

    // 3. Build a k-d tree for neighborhood search
    pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>());
    kdtree->setInputCloud(cloud_in);

    // 4. Create ISS keypoint detector and set parameters
    pcl::ISSKeypoint3D<PointT, PointT> iss_detector;
    iss_detector.setSearchMethod(kdtree);
    iss_detector.setInputCloud(cloud_in);

    // ---- Adjust the following parameters based on actual cloud size/resolution ----
    double model_resolution = .28;  // Approximate resolution, adjust for your data

    // salient_radius is usually 6~8 times model_resolution
    iss_detector.setSalientRadius(4* model_resolution);

    // non_max_radius is usually 4~5 times model_resolution
    iss_detector.setNonMaxRadius(1 * model_resolution);

    // These two thresholds determine filtering standards for the covariance matrix eigenvalue ratios
    iss_detector.setThreshold21(0.999);
    iss_detector.setThreshold32(0.999);

    // Minimum number of neighbors in the neighborhood (exclude overly sparse areas)
    iss_detector.setMinNeighbors(5);

    // In PCL 1.10, there is no setNormalsEstimationMethod interface, so remove that line
    // iss_detector.setNormalsEstimationMethod(...); // Not present in PCL 1.10

    // 5. Execute keypoint detection
    iss_detector.compute(*keypoints_out);

    std::cout << "ISS keypoint detection completed, detected keypoints: "
              << keypoints_out->size() << std::endl;

    // 6. Save the result
    pcl::io::savePCDFileASCII(output_pcd, *keypoints_out);
    std::cout << "Keypoints saved to: " << output_pcd << std::endl;


    // ------------------------------------------------------
    // 9. Simple visualization: show the original cloud (white) + keypoints (red)
    // ------------------------------------------------------
    
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Keypoints Viewer"));
    viewer->setBackgroundColor(0, 0, 0);  // Optional: set background to black

    // Add the original cloud (default white)
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_color_handler(cloud_in, 255, 255, 255); 
    viewer->addPointCloud<PointT>(cloud_in, cloud_color_handler, "cloud_in");

    // Add keypoints (red), and set a larger point size for better visibility
    pcl::visualization::PointCloudColorHandlerCustom<PointT> keypoints_color_handler(keypoints_out, 255, 0, 0);
    viewer->addPointCloud<PointT>(keypoints_out, keypoints_color_handler, "keypoints_out");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "keypoints_out");

    // Camera parameters can be adjusted as needed; here we simply let it auto-fit
    viewer->addCoordinateSystem(0.1);  // Optional: display coordinate axes
    viewer->initCameraParameters();

    // Loop so the window remains open until manually closed
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
   
    return 0;
}
