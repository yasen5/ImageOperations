#include "SimpleEdgeDetection/edge_detector.h"
#include <opencv2/opencv.hpp>

using namespace edge_detection;
using MatrixXfRM = Matrix<float, Dynamic, Dynamic, RowMajor>;

int main() {
  const std::string img_name = "test_images/test_image.png";
  const cv::Mat image =
      cv::imread("/Users/yasen/CLionProjects/ImageGradients/" + img_name,
                 cv::IMREAD_COLOR);

  // Apply kernels
  Matrix<uint8_t, Dynamic, Dynamic, RowMajor> kerneled1 =
      CleanupEdges(ApplyKernel(image, ROBERTS), 0.3, 0.7);
  cv::Mat cv_kerneled1(kerneled1.rows(), kerneled1.cols(), CV_8UC1,
                       kerneled1.data());

  Matrix<uint8_t, Dynamic, Dynamic, RowMajor> kerneled2 =
      CleanupEdges(ApplyKernel(image, SOBEL3x3), 0.3, 0.4);
  cv::Mat cv_kerneled2(kerneled2.rows(), kerneled2.cols(), CV_8UC1,
                       kerneled2.data());

  Matrix<uint8_t, Dynamic, Dynamic, RowMajor> kerneled3 =
      CleanupEdges(ApplyKernel(image, SOBEL5x5), 0.3, 0.7);
  cv::Mat cv_kerneled3(kerneled3.rows(), kerneled3.cols(), CV_8UC1,
                       kerneled3.data());

  cv::resize(cv_kerneled2, cv_kerneled2, cv_kerneled1.size());
  cv::resize(cv_kerneled3, cv_kerneled3, cv_kerneled1.size());

  cv::Mat combined;
  cv::hconcat(std::vector<cv::Mat>{cv_kerneled1, cv_kerneled2, cv_kerneled3},
              combined);
  cv::imshow("All Kernels Side-by-Side", combined);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}