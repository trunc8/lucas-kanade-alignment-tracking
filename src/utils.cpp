#include "utils.h"

namespace utilities
{
    void visualizeInputs()
    {
        cv::Mat image;
        std::stringstream ss;

        for (size_t img_idx = 0; img_idx < constants::num_frames; img_idx++)
        {
            ss.str("");
            ss.width(3);
            ss.fill('0');
            ss << img_idx + 1;
            image = cv::imread(constants::data_path + "/00000" + ss.str() + ".jpg", cv::IMREAD_COLOR);

            if (!image.data)
            {
                std::cout << "No image data" << std::endl;
                return;
            }

            cv::namedWindow("Visualizing Data Frames", cv::WINDOW_AUTOSIZE);
            cv::imshow("Visualizing Data Frames", image);

            cv::waitKey(10);
        }
        cv::destroyAllWindows();
    }
    
    std::vector<bool> readOcclusions()
    {
        std::vector<bool> occlusions(constants::num_frames, false);
        std::ifstream occlusion_file;
        std::string occlusion_filename = constants::data_path + "/occlusion.label";
        occlusion_file.open(occlusion_filename);

        if (!occlusion_file.is_open())
        {
            std::cout << "occlusions file not found" << std::endl;
            return occlusions;
        }

        std::cout << occlusion_filename + " is opened" << std::endl;

        for (size_t img_idx = 0; img_idx < constants::num_frames; img_idx++)
        {
            // Each line contains one bool: occluded
            std::string str_val;
            getline(occlusion_file, str_val);
            occlusions[img_idx] = std::stoi(str_val) == 1;
        }

        return occlusions;
    }

    std::vector<std::vector<int>> readGroundtruths()
    {
        std::vector<std::vector<int>> groundtruths(constants::num_frames, std::vector<int>(8, 0));
        std::ifstream groundtruth_file;
        std::string groundtruth_filename = constants::data_path + "/groundtruth.txt";
        groundtruth_file.open(groundtruth_filename);

        if (!groundtruth_file.is_open())
        {
            std::cout << "groundtruths file not found" << std::endl;
            return groundtruths;
        }

        std::cout << groundtruth_filename + " is opened" << std::endl;

        for (size_t img_idx = 0; img_idx < constants::num_frames; img_idx++)
        {
            // x1, y1, x2, y2, x3, y3, x4, y4
            std::string line;
            getline(groundtruth_file, line);
            std::istringstream iline(line);
            for (size_t i = 0; i < 8; i++)
            {
                std::string str_val;
                getline(iline, str_val, ',');
                groundtruths[img_idx][i] = std::stoi(str_val);
            }
        }

        return groundtruths;
    }

    void visualizeGroundtruths(const std::vector<std::vector<int>> &groundtruths)
    {
        cv::Mat image;
        std::stringstream ss;

        for (size_t img_idx = 0; img_idx < constants::num_frames; img_idx++)
        {
            ss.str("");
            ss.width(3);
            ss.fill('0');
            ss << img_idx + 1;
            image = cv::imread(constants::data_path + "/00000" + ss.str() + ".jpg", cv::IMREAD_COLOR);

            if (!image.data)
            {
                std::cout << "No image data! Returning\n\n" << std::endl;
                return;
            }
            int x2 = groundtruths[img_idx][2];
            int y2 = groundtruths[img_idx][3];
            int x4 = groundtruths[img_idx][6];
            int y4 = groundtruths[img_idx][7];
            cv::Point2i org{x2, y2};
            cv::Size2i sz(x4 - x2, y4 - y2);

            cv::rectangle(image, cv::Rect(org, sz), cv::Scalar(0, 255, 0, 255));

            cv::namedWindow("Visualizing Groundtruth Bounding Boxes", cv::WINDOW_AUTOSIZE);
            cv::imshow("Visualizing Groundtruth Bounding Boxes", image);

            cv::waitKey(10);
        }
        cv::destroyAllWindows();
    }
}