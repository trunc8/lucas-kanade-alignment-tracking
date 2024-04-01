#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp> // Needs to be after eigen3 include
#include <opencv2/opencv.hpp>

#include "constants.h"

namespace utilities
{
    void visualizeInputs();

    std::vector<bool> readOcclusions();

    std::vector<std::vector<int>> readGroundtruths();

    void visualizeGroundtruths(const std::vector<std::vector<int>> &groundtruths);
}