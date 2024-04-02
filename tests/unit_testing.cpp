#include <gtest/gtest.h>
#include <tracker.h>

TEST(TrackerTest, UseMomentumOnOcclusionTest)
{
    /*
     * Useful constant variables for this test
    */
    Eigen::MatrixXf M_zeros = Eigen::MatrixXf::Constant(2, 3, 0);
    Eigen::MatrixXf M_identity = Eigen::MatrixXf::Identity(2, 3);

    /*
     * Variables needed to run test function
    */
    Tracker tracker_obj;
    bool is_occluded;
    Eigen::MatrixXf M;
    int img_idx;    

    /*
     * Test 1
    */
    is_occluded = false;
    img_idx = 0;
    M = M_zeros; // performs deep copy
    tracker_obj.useMomentumOnOcclusion(is_occluded, M, img_idx);
    EXPECT_EQ(M, M_zeros);

    /*
     * Test 2
     * This will only work when tracker_obj.m_velocities 
     * are default initialized as 0's
    */
    is_occluded = true;
    img_idx = constants::num_frames_for_momentum;
    tracker_obj.useMomentumOnOcclusion(is_occluded, M, img_idx);
    EXPECT_EQ(M, M_identity);
}

TEST(TrackerTest, PerformLucasKanadeAffineTest)
{
    /*
     * Useful constant variables for this test
    */
    Eigen::MatrixXf M_identity = Eigen::MatrixXf::Identity(2, 3);
    Eigen::MatrixXf M_test2(2, 3);
    M_test2 << 1.00231, -0.00243187, 0, 
    0.000864921, 0.999124, 0;

    /*
     * Variables needed to run test function
    */
    Tracker tracker_obj;
    cv::Mat It;
    cv::Mat It1;
    Eigen::RowVectorXf rect(4);
    Eigen::MatrixXf M;

    // /*
    //  * Test 1
    // */
    It = cv::Mat::zeros(10, 10, CV_64F);
    It1 = cv::Mat::zeros(10, 10, CV_64F);
    rect << 1, 1, 4, 4;
    M = tracker_obj.performLucasKanadeAffine(It, It1, rect.cast<int>());
    EXPECT_EQ(M, M_identity);

    /*
     * Test 2
    */
    It = cv::Mat::zeros(100, 100, CV_64F);
    It(cv::Rect(50, 50, 4, 8)) = 1;
    It1 = cv::Mat::zeros(100, 100, CV_64F);
    It1(cv::Rect(52, 50, 4, 8)) = 1;
    rect << 50, 50, 53, 50;
    M = tracker_obj.performLucasKanadeAffine(It, It1, rect.cast<int>());
    ASSERT_TRUE(M.isApprox(M_test2));
}