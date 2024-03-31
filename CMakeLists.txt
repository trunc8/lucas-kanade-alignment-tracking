cmake_minimum_required(VERSION 3.2)
project(CarDetectionProject)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g")

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(main src/main.cpp src/utils.cpp)
target_link_libraries(main ${OpenCV_LIBS} Eigen3::Eigen)
target_include_directories(main PRIVATE include)