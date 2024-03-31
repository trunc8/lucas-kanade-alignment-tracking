#include "utils.h"

void visualizeInputs()
{
    cv::Mat image;
    std::stringstream ss;

    for (int i = 0; i < NUM_IMAGES; i++)
    {
        ss.str("");
        ss.width(3);
        ss.fill('0');
        ss << i + 1;
        image = cv::imread("../data/00000" + ss.str() + ".jpg", cv::IMREAD_COLOR);

        if (!image.data)
        {
            std::cout << "No image data" << std::endl;
            return;
        }

        cv::namedWindow("Display image", cv::WINDOW_AUTOSIZE);
        cv::imshow("Display image", image);

        cv::waitKey(1);
    }
}

std::vector<bool> readOcclusions()
{
    std::vector<bool> occlusions(NUM_IMAGES, false);
    std::ifstream occlusion_file;
    std::string occlusion_filename = "../data/occlusion.label";
    occlusion_file.open(occlusion_filename);

    if (!occlusion_file.is_open())
    {
        std::cout << "occlusions file not found" << std::endl;
        return occlusions;
    }

    std::cout << occlusion_filename + " is opened" << std::endl;

    for (int img_idx = 0; img_idx < NUM_IMAGES; img_idx++)
    {
        // Each line contains one bool: occluded
        std::string str_val;
        getline(occlusion_file, str_val);
        occlusions[img_idx] = std::stoi(str_val) == 1;
    }

    // for (auto i : occlusions)
    // {
    //     std::cout << i << "\t";
    // }
    // std::cout << std::endl;

    return occlusions;
}

std::vector<std::vector<int>> readGroundtruths()
{
    std::vector<std::vector<int>> groundtruths(NUM_IMAGES, std::vector<int>(8, 0));
    std::ifstream groundtruth_file;
    std::string groundtruth_filename = "../data/groundtruth.txt";
    groundtruth_file.open(groundtruth_filename);

    if (!groundtruth_file.is_open())
    {
        std::cout << "groundtruths file not found" << std::endl;
        return groundtruths;
    }

    std::cout << groundtruth_filename + " is opened" << std::endl;

    for (int img_idx = 0; img_idx < NUM_IMAGES; img_idx++)
    {
        // x1, y1, x2, y2, x3, y3, x4, y4
        std::string line;
        getline(groundtruth_file, line);
        std::istringstream iline(line);
        for (int i = 0; i < 8; i++)
        {
            std::string str_val;
            getline(iline, str_val, ',');
            groundtruths[img_idx][i] = std::stoi(str_val);
        }
    }

    // for (auto v : groundtruths)
    // {
    //     for (auto i : v)
    //     {
    //         std::cout << i << "\t";
    //     }
    //     std::cout << std::endl;
    // }

    return groundtruths;
}

void visualizeGroundtruths(std::vector<std::vector<int>> groundtruths)
{
    cv::Mat image;
    std::stringstream ss;

    for (int i = 0; i < NUM_IMAGES; i++)
    {
        ss.str("");
        ss.width(3);
        ss.fill('0');
        ss << i + 1;
        image = cv::imread("../data/00000" + ss.str() + ".jpg", cv::IMREAD_COLOR);

        if (!image.data)
        {
            std::cout << "No image data" << std::endl;
            return;
        }
        int x2 = groundtruths[i][2];
        int y2 = groundtruths[i][3];
        int x4 = groundtruths[i][6];
        int y4 = groundtruths[i][7];
        cv::Point2i org{x2, y2};
        cv::Size2i sz(x4 - x2, y4 - y2);

        cv::rectangle(image, cv::Rect(org, sz), cv::Scalar(0, 255, 0, 255));

        cv::namedWindow("Display image", cv::WINDOW_AUTOSIZE);
        cv::imshow("Display image", image);

        cv::waitKey(1);
    }
}