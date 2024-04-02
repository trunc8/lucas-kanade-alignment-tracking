#pragma once

#include <cstdint>

namespace constants
{
    const std::string data_path = "../data";
    constexpr size_t num_frames = 252;
    constexpr size_t num_frames_for_momentum = 4;
    constexpr size_t max_iterations_LKA = 100;
    constexpr float dp_threshold_LKA = 0.02;
}