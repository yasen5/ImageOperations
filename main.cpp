#include "SimpleEdgeDetection/edge_detector.h"
#include <opencv2/opencv.hpp>

int main() {
    const std::string img_name = "test_images/test_image3.png";
    cv::Mat image = cv::imread("/Users/yasen/CLionProjects/ImageGradients/" + img_name, cv::IMREAD_COLOR_BGR);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    // image.convertTo(image, CV_32FC1, 1.0/255.0);
    Map<Matrix<uint8_t, Dynamic, Dynamic, RowMajor>> eigen_mat(image.ptr<uint8_t>(), image.rows, image.cols);
    cv::Mat converted(eigen_mat.rows(), eigen_mat.cols(), CV_8U, eigen_mat.data());
    std::cout << "Channels: " << converted.channels() << std::endl;
    cv::imshow("Edges", converted);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}