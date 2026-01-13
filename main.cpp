#include "SimpleEdgeDetection/edge_detector.h"
#include <opencv2/opencv.hpp>

using namespace edge_detection;
using MatrixXfRM = Matrix<float, Dynamic, Dynamic, RowMajor>;

constexpr size_t DISPLAY_SIZE_X = 500;
constexpr size_t DISPLAY_SIZE_Y = 500;

cv::Mat get_kerneled(const cv::Mat &image, Kernel kernel_type,
                     const bool edge) {
  const MatrixXf kerneled = ApplyKernel(image, kernel_type, 1);
  if (edge) {
    Matrix<uint8_t, Dynamic, Dynamic, RowMajor> cleaned =
        CleanupEdges(kerneled, 0.3, 0.4);
    const cv::Mat cv_kerneled(cleaned.rows(), cleaned.cols(), CV_8UC1,
                              cleaned.data());
    return cv_kerneled.clone();
  } else {
    MatrixXfRM row_major = kerneled;
    const cv::Mat cv_kerneled(row_major.rows(), row_major.cols(), CV_32FC1,
                              row_major.data());
    return cv_kerneled.clone();
  }
}

void resize(std::vector<cv::Mat> &img_list) {
  for (cv::Mat &img : img_list) {
    cv::resize(img, img, cv::Size2d{DISPLAY_SIZE_X, DISPLAY_SIZE_Y});
  }
}

int main() {
  const std::string img_name = "test_images/test_image.png";
  const cv::Mat image =
      cv::imread("/Users/yasen/CLionProjects/ImageGradients/" + img_name,
                 cv::IMREAD_COLOR);

  const std::vector<Kernel> edge_kernels = {ROBERTS, SOBEL3x3, SOBEL5x5};
  const std::vector<Kernel> other_kernels = {GAUSSIAN3x3, GAUSSIAN5x5};
  std::vector<cv::Mat> edge_detected;
  std::vector<cv::Mat> other;
  for (const Kernel kernel : edge_kernels) {
    edge_detected.push_back(get_kerneled(image, kernel, true));
  }
  for (const Kernel kernel : other_kernels) {
    other.push_back(get_kerneled(image, kernel, false));
  }

  resize(edge_detected);
  resize(other);
  cv::Mat combined_edge;
  cv::hconcat(edge_detected, combined_edge);
  cv::imshow("Edge Kernels", combined_edge);
  cv::Mat combined_other;
  cv::hconcat(other, combined_other);
  cv::imshow("Other Kernels", combined_other);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}