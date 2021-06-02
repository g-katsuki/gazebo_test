cmake_minimum_required(VERSION 3.5)
project(potential_angle)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(rclcpp_components REQUIRED)
set(NODES potential_angle)
#add_executable(${target} src/${target}.cpp)
#ament_target_dependencies(${target} rclcpp std_msgs geometry_msgs sensor_msgs rclcpp_components)
#install(TARGETS ${target} DESTINATION lib/${PROJECT_NAME})
#endforeach()

add_library(${PROJECT_NAME} SHARED
  src/potential_angle.cpp
)
ament_target_dependencies(${PROJECT_NAME} rclcpp std_msgs geometry_msgs sensor_msgs nav_msgs tf2 rclcpp_components)
target_link_libraries(${PROJECT_NAME} rs_navi rs_mapping ${Boost_LIBRARIES})
rclcpp_components_register_nodes(${PROJECT_NAME} "potential::PotentialAngle")

install(TARGETS ${PROJECT_NAME} DESTINATION lib)

ament_package()