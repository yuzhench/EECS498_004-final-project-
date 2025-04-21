cmake_minimum_required(VERSION 3.10)
project(harris_3d_demo)

# find_package(PCL 1.10 REQUIRED COMPONENTS io visualization keypoints)

# include_directories(${PCL_INCLUDE_DIRS})
# link_directories(${PCL_LIBRARY_DIRS})
# add_definitions(${PCL_DEFINITIONS})


find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(harris_3d_demo keypointPCL.cpp)
target_link_libraries(harris_3d_demo ${PCL_LIBRARIES})
