cmake_minimum_required(VERSION 3.8)
project(probl_m)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include)

add_executable(kf_node src/kalman_filter_node.cpp)

ament_target_dependencies(kf_node
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2
  tf2_geometry_msgs
)

target_link_libraries(kf_node Eigen3::Eigen)

add_executable(ekf_node src/extended_kalman_filter_node.cpp)

ament_target_dependencies(ekf_node
  rclcpp
  geometry_msgs
  nav_msgs
  tf2
  tf2_geometry_msgs
)

target_link_libraries(ekf_node Eigen3::Eigen)

add_executable(pf_node src/particle_filter_node.cpp)

ament_target_dependencies(pf_node
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2
  tf2_geometry_msgs
)

target_link_libraries(pf_node Eigen3::Eigen)

add_executable(evaluator_node src/evaluator_node.cpp)

ament_target_dependencies(evaluator_node
  rclcpp
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

add_executable(initial_pose_node src/initial_pose_node.cpp)

ament_target_dependencies(initial_pose_node
  rclcpp
  geometry_msgs
)

add_executable(waypoint_node src/waypoint_node.cpp)

ament_target_dependencies(waypoint_node
  rclcpp
  rclcpp_action
  nav2_msgs
  geometry_msgs
)

target_link_libraries(evaluator_node Eigen3::Eigen)

install(TARGETS
  initial_pose_node
  waypoint_node
  kf_node
  ekf_node
  pf_node
  evaluator_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()