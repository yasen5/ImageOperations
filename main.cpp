#include "SimpleEdgeDetection/edge_detector.h"
#include <opencv2/opencv.hpp>

using namespace edge_detection;
using MatrixXfRM = Matrix<float, Dynamic, Dynamic, RowMajor>;

cv::Mat get_kerneled(const cv::Mat &image, Kernel kernel_type) {
  Matrix<uint8_t, Dynamic, Dynamic, RowMajor> kerneled =
      CleanupEdges(ApplyKernel(image, kernel_type), 0.3, 0.4);
  const cv::Mat cv_kerneled(kerneled.rows(), kerneled.cols(), CV_8UC1,
                      kerneled.data());
  return cv_kerneled.clone();
}

void resize(std::vector<cv::Mat> &img_list) {
  for (cv::Mat &img : img_list) {
    cv::resize(img, img, img_list[0].size());
  }
}

int main() {
  const std::string img_name = "test_images/test_image.png";
  const cv::Mat image =
      cv::imread("/Users/yasen/CLionProjects/ImageGradients/" + img_name,
                 cv::IMREAD_COLOR);

  std::vector<cv::Mat> kerneled{get_kerneled(image, ROBERTS),
                                get_kerneled(image, SOBEL3x3),
                                get_kerneled(image, SOBEL5x5)};
  resize(kerneled);
  cv::Mat combined;
  cv::hconcat(kerneled, combined);
  cv::imshow("All Kernels Side-by-Side", combined);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}