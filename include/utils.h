#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "constants.h"

namespace utilities
{
    /**
     * @brief Visualizes all images in the data folder
     * 
     * @details Depends on two constants: num_frames and data_path
     * Update these constants in constants.h file if using different data
     * 
     */
    void visualizeInputs();

    /**
     * @brief Reads occlusion flags for each frame from occlusion.label
     * 
     * @details Depends on two constants: num_frames and data_path
     * Update these constants in constants.h file if using different data.
     * Assumes file name is occlusion.label
     * 
     * @return std::vector<bool> occlusions: one flag for each frame
     */
    std::vector<bool> readOcclusions();

    /**
     * @brief Read the groundtruth bounding box for each frame from groundtruth.txt
     * 
     * @details Depends on two constants: num_frames and data_path
     * Update these constants in constants.h file if using different data
     * Assumes file name is groundtruth.txt
     * 
     * @return std::vector<std::vector<int>> groundtruths: 8 integers for each frame
     */
    std::vector<std::vector<int>> readGroundtruths();

    /**
     * @brief Visualize the groundtruth bounding boxes for each image in the data folder
     * 
     * @details Depends on two constants: num_frames and data_path
     * Update these constants in constants.h file if using different data
     * 
     */
    void visualizeGroundtruths(const std::vector<std::vector<int>> &groundtruths);
}