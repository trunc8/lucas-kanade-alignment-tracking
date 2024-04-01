#include "utils.h"


Eigen::MatrixXf performLucasKanadeAffine(const cv::Mat &It_opencv, const cv::Mat &It1_opencv, const Eigen::RowVectorXi &rect);

void useMomentumOnOcclusion(bool is_occluded, Eigen::MatrixXf &M, Eigen::MatrixXf velocities, int img_idx);

void performTracking();