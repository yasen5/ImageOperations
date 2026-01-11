#include "SimpleEdgeDetection/edge_detector.h"
#include <opencv2/opencv.hpp>

using MatrixXfRM = Matrix<float, Dynamic, Dynamic, RowMajor>;

int main() {
  const std::string img_name = "test_images/test_image.png";
  const cv::Mat image =
      cv::imread("/Users/yasen/CLionProjects/ImageGradients/" + img_name,
                 cv::IMREAD_COLOR);
  Matrix<uint8_t, Dynamic, Dynamic, RowMajor> kerneled =
      EdgeDetector::CleanupEdges(EdgeDetector::ApplyKernel(image, SOBEL), 0.3,
                                 0.7);
  const cv::Mat cv_kerneled(kerneled.rows(), kerneled.cols(), CV_8UC1,
                            kerneled.data());
  cv::imshow("Edges", cv_kerneled);
  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}