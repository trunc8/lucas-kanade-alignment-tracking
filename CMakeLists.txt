cmake_minimum_required(VERSION 3.2)
project(CarDetectionProject)

#-------------------------------------------------------------------
# Build type. Default is Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message("Build type: " ${CMAKE_BUILD_TYPE})

#-------------------------------------------------------------------
# C++ flags and standard. Using C++20
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 20)

#-------------------------------------------------------------------
# Executables should be created in bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

#-------------------------------------------------------------------
# Package dependences. If these fail, check README.md for installation
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

#-------------------------------------------------------------------
# Installing gtest
# Source: https://google.github.io/googletest/quickstart-cmake.html
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

#-------------------------------------------------------------------
# Define executables. Only one in our case.
add_executable(
    main
    src/main.cpp
    src/tracker.cpp
    src/utils.cpp
)
target_link_libraries(
    main
    ${OpenCV_LIBS}
    Eigen3::Eigen
)
target_include_directories(main PRIVATE include)

#-------------------------------------------------------------------
# Unit testing with GTest
enable_testing()

add_executable(
    unit_testing
    tests/unit_testing.cpp
    src/tracker.cpp 
    src/utils.cpp
)
target_link_libraries(
    unit_testing
    GTest::gtest_main
    ${OpenCV_LIBS}
    Eigen3::Eigen
)
target_include_directories(unit_testing PRIVATE include)

include(GoogleTest)
gtest_discover_tests(unit_testing)
#-------------------------------------------------------------------