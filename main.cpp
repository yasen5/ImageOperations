#include "SimpleEdgeDetection/edge_detector.h"
#include <opencv2/opencv.hpp>

int main() {
    const std::string img_name = "test_images/test_image3.png";
    const cv::Mat image = cv::imread("/Users/yasen/CLionProjects/ImageGradients/" + img_name, cv::IMREAD_GRAYSCALE);
    cv::Mat edges;
    MatrixXf eigen_mat = EdgeDetector::ApplyKernel(image, GAUSSIAN, 1);
    cv::eigen2cv(eigen_mat, edges);
    cv::imshow("Edges", edges);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}