#include "tracker.h"

Eigen::MatrixXf performLucasKanadeAffine(const cv::Mat &It_opencv, const cv::Mat &It1_opencv, const Eigen::RowVectorXi &rect)
{
    Eigen::MatrixXf M = Eigen::MatrixXf::Identity(2, 3);
    Eigen::MatrixXf M_inv(2, 3);
    cv::Mat M_inv_opencv(2, 3, CV_64F);

    int x1 = rect(0);
    int y1 = rect(1);
    int x2 = rect(2);
    int y2 = rect(3);

    // Prepare bbox info of It image, which is called the template
    int bbox_W = x2 - x1 + 1;
    int bbox_H = y2 - y1 + 1;

    Eigen::VectorXf bbox_Xs = Eigen::VectorXf::LinSpaced(bbox_W, x1, x2);
    Eigen::VectorXf bbox_Ys = Eigen::VectorXf::LinSpaced(bbox_H, y1, y2);

    Eigen::VectorXf flat_bbox_Xs = bbox_Xs.replicate(bbox_Ys.size(), 1);
    Eigen::VectorXf flat_bbox_Ys = bbox_Ys.replicate(1, bbox_Xs.size()).transpose().reshaped();

    // Extract template from image It
    Eigen::MatrixXf It;
    cv::cv2eigen(It_opencv, It);
    Eigen::VectorXf flat_template = It.block(y1, x1, bbox_H, bbox_W).reshaped<Eigen::RowMajor>();

    // gradients of It1
    cv::Mat It1x_opencv, It1y_opencv;
    cv::Sobel(It1_opencv, It1x_opencv, CV_64F, 1, 0, 1, 0.5); // x-gradient
    cv::Sobel(It1_opencv, It1y_opencv, CV_64F, 0, 1, 1, 0.5); // y-gradient

    // Create placeholder variables
    cv::Mat warped_target_opencv, gradI_xp_opencv, gradI_yp_opencv;
    Eigen::MatrixXf warped_target, gradI_xp, gradI_yp;
    Eigen::VectorXf flat_gradI_xp, flat_gradI_yp;

    Eigen::ArrayXf column_arr(bbox_W * bbox_H);

    // dp: Change in the parameters in M
    // It is in the form of dp1 dp2 dp3 dp4 dp5 dp6
    Eigen::VectorXf dp;

    for (int itr = 0; itr < maxIterations; itr++) // TODO: Make it maxIterations again
    {
        // 1. Warp Image
        M_inv.block(0, 0, 2, 2) = M.block(0, 0, 2, 2).inverse();
        M_inv.col(2) = -M.col(2);
        cv::eigen2cv(M_inv, M_inv_opencv);

        cv::warpAffine(It1_opencv, warped_target_opencv, M_inv_opencv, It1_opencv.size());
        cv::warpAffine(It1x_opencv, gradI_xp_opencv, M_inv_opencv, It_opencv.size());
        cv::warpAffine(It1y_opencv, gradI_yp_opencv, M_inv_opencv, It_opencv.size());

        cv::cv2eigen(warped_target_opencv, warped_target);
        cv::cv2eigen(gradI_xp_opencv, gradI_xp);
        cv::cv2eigen(gradI_yp_opencv, gradI_yp);

        Eigen::VectorXf flat_warped_target = warped_target.block(y1, x1, bbox_H, bbox_W).reshaped<Eigen::RowMajor>();

        // 2. Compute Error [y1 : (y2 + 1), x1 : (x2 + 1)]
        Eigen::VectorXf b = flat_template - flat_warped_target;

        // 3. Compute Gradient
        flat_gradI_xp = gradI_xp.block(y1, x1, bbox_H, bbox_W).reshaped<Eigen::RowMajor>();
        flat_gradI_yp = gradI_yp.block(y1, x1, bbox_H, bbox_W).reshaped<Eigen::RowMajor>();

        // 4. Evaluate Jacobian to construct matrix A to solve non-linear least sq Ax~b
        /***
         * A = np.stack(
            [
                gradI_xp * XV_flat,
                gradI_yp * XV_flat,
                gradI_xp * YV_flat,
                gradI_yp * YV_flat,
                gradI_xp,
                gradI_yp,
            ],
            axis=1,
        )
        */
        Eigen::MatrixXf A(bbox_W * bbox_H, 6);

        A.col(0) = flat_gradI_xp.array() * flat_bbox_Xs.array();

        A.col(1) = flat_gradI_yp.array() * flat_bbox_Xs.array();

        A.col(2) = flat_gradI_xp.array() * flat_bbox_Ys.array();

        A.col(3) = flat_gradI_yp.array() * flat_bbox_Ys.array();

        A.block(0, 4, bbox_W * bbox_H, 2) << flat_gradI_xp, flat_gradI_yp; // columns 5 and 6

        // 6. Compute dp and update M
        dp = (A.transpose() * A).ldlt().solve(A.transpose() * b);
        M += dp.reshaped(2, 3);

        if (dp.norm() < thresh)
        {
            // std::cout << "Converged at " << itr << std::endl;
            break;
        }
    }

    return M;
}

void useMomentumOnOcclusion(bool is_occluded, Eigen::MatrixXf &M, Eigen::MatrixXf velocities, int img_idx)
{
    if (is_occluded)
    {
        assert(img_idx >= 2); // Doesn't handle if object occluded in the first two frames
        Eigen::MatrixXf frame_velocity = velocities.block(img_idx, 0, 2, 2).colwise().mean();
        frame_velocity(0, 1) = -frame_velocity(0, 1);
        M.block(0, 0, 2, 2) = Eigen::Matrix2f::Identity();
        M.col(2) = frame_velocity.transpose();
    }
}

void performTracking()
{
    std::vector<bool> occlusions = utilities::readOcclusions();

    Eigen::MatrixXf velocities(NUM_IMAGES - 1, 2);

    Eigen::RowVectorXf initial(4);
    initial << 6, 166, 49, 193; // Opencv conventions

    /* Source: https://ros-developer.com/2018/12/23/camera-projection-matrix-with-eigen/
    OpenCV coordinate system
                      ▲
                     /
                    /
                  Z/
                  /
                 /
    ------------------------------► X
                |
                |
                |
                |
              Y |
                ▼
    */

    Eigen::MatrixXf rects(NUM_IMAGES, 4);
    rects.row(0) = initial;

    Eigen::MatrixXf M(2, 3);       // Affine matrix
    Eigen::MatrixXf corners(3, 2); // Homogenous coordinates
    Eigen::RowVectorXf rect, new_rect;
    Eigen::RowVectorXf frame_velocity;
    cv::Mat It_color, It, It1;
    std::stringstream ss;

    std::cout << std::setprecision(2) << std::fixed;

    for (int img_idx = 0; img_idx < NUM_IMAGES - 1; img_idx++)
    {
        ss.str("");
        ss.width(3);
        ss.fill('0');
        ss << img_idx + 1;
        std::string img0_filename = "../data/00000" + ss.str() + ".jpg";

        ss.str("");
        ss.width(3);
        ss.fill('0');
        ss << img_idx + 2;
        std::string img1_filename = "../data/00000" + ss.str() + ".jpg";

        It_color = cv::imread(img0_filename, cv::IMREAD_COLOR);
        It = cv::imread(img0_filename, cv::IMREAD_GRAYSCALE);
        It1 = cv::imread(img1_filename, cv::IMREAD_GRAYSCALE);

        rect = rects.row(img_idx);

        /***
         * Run algorithm and collect rects
         */
        M = performLucasKanadeAffine(It, It1, rect.cast<int>());

        /***
         * Momentum update
         */
        bool is_occluded = occlusions[img_idx];
        useMomentumOnOcclusion(is_occluded, M, velocities, img_idx);

        frame_velocity = M.col(2).transpose();
        velocities.row(img_idx) = frame_velocity;

        corners.col(0) << rect(0), rect(1), 1;
        corners.col(1) << rect(2), rect(3), 1;

        new_rect = (M * corners).reshaped(1, 4); // **interpreted in column-major**;
        rects.row(img_idx + 1) = new_rect;

        std::cout << "Frame " << img_idx + 1 << " |";
        std::cout << "\tRect: " << new_rect.cast<int>();
        std::cout << "\tVelocity: " << frame_velocity;
        std::cout << std::endl;

        int x1 = new_rect.cast<int>()(0);
        int y1 = new_rect.cast<int>()(1);
        int x2 = new_rect.cast<int>()(2);
        int y2 = new_rect.cast<int>()(3);

        cv::Point2i org{x1, y1};
        cv::Size2i sz(x2 - x1, y2 - y1);

        cv::rectangle(It_color, cv::Rect(org, sz), cv::Scalar(0, 255, 0, 255));

        cv::namedWindow("Display image", cv::WINDOW_AUTOSIZE);
        cv::imshow("Display image", It_color);

        cv::waitKey(1);
    }
}