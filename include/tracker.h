#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp> // Needs to be after eigen3 include

#include "utils.h"

class Tracker
{
private:
    std::vector<bool> m_occlusions;
    Eigen::MatrixXf m_velocities;
    Eigen::RowVectorXf m_initial_bounding_box;
    Eigen::MatrixXf m_rects;
    bool m_to_viz_groundtruth_on_tracking;
    std::vector<std::vector<int>> m_groundtruths;

public:
    /**
     * @brief Construct a new Tracker object
     *
     * @details This is a delegated constructor. It runs parameterized constructor with
     * blank values.
     * It takes care to input the correct dimensions for  groundtruths, else the
     * intersection over union calculation will segfault.
     *
     */
    Tracker();

    /**
     * @brief Construct a new Tracker object
     *
     * @details Reads and stores occlusion data. Initializes member variables
     *
     * @param to_viz_groundtruth_on_tracking: flag whether to overlay groundtruth bboxes
     * on tracking visualization
     * @param groundtruths: groundtruth is required to calculate the intersection over
     * union and visualize the
     * overlay if previous flag is set
     */
    Tracker(bool to_viz_groundtruth_on_tracking, std::vector<std::vector<int>> groundtruths);

    /**
     * @brief Implements the Lucas Kanade Image Alignment algorithm
     *
     * @details Detailed description. More detail.
     * @see https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method
     *
     * @param It_opencv: Entire image in current frame. Contains the template patch.
     * @param It1_opencv: Entire image next frame. Serves as the warped target.
     * @param rect: Bounding box of template patch
     * @return Eigen::MatrixXf M: Affine transformation matrix
     */
    Eigen::MatrixXf performLucasKanadeAffine(const cv::Mat &It_opencv, const cv::Mat &It1_opencv,
                                             const Eigen::RowVectorXi &rect);

    /**
     * @brief Performs momentum based tracking when template is occluded
     *
     * @details Uses one constant num_frames_for_momentum which is a tunable parameter.
     * Removes rotation and scaling, and assumes only translation
     * Translation calculated using average velocity over last "num_frames_for_momentum" frames.
     *
     * @param is_occluded: groundtruth label if the template is occluded.
     * TODO: Infer is_occluded from image frames itself
     * @param M: Affine transformation matrix calculated by LKA
     * @param img_idx: Image frame number used to query velocities table
     */
    void useMomentumOnOcclusion(bool is_occluded, Eigen::MatrixXf &M, size_t img_idx);

    /**
     * @brief Root function that orchestrates the entire tracking
     *
     * @details Iterates over the data images, calls LKA function,
     * calculates statistics, and visualizes output.
     *
     */
    void performTracking();
};