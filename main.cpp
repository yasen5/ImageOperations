#include "SimpleEdgeDetection/canny.h"
#include "SimpleEdgeDetection/kernel.h"
#include <opencv2/opencv.hpp>

using namespace edge_detection;

constexpr size_t DISPLAY_SIZE_X = 500;
constexpr size_t DISPLAY_SIZE_Y = 500;

cv::Mat_<float> get_kerneled(const cv::Mat_<float> &image, Kernel kernel_type,
                             const bool edge) {
  cv::Mat_<float> kerneled = abs(ApplyKernel(image, kernel_type, 1));
  if (edge) {
    kerneled += abs(ApplyKernel(image, kernel_type, 1, 0, true));
  }
  return kerneled;
}

void resize(std::vector<cv::Mat_<float>> &img_list) {
  for (cv::Mat_<float> &img : img_list) {
    cv::resize(img, img, cv::Size2d{DISPLAY_SIZE_X, DISPLAY_SIZE_Y});
  }
}

int main() {
  const std::string img_name = "test_images/test_image4.jpg";
  const cv::Mat color_image =
      cv::imread("/Users/yasen/CLionProjects/ImageGradients/" + img_name,
                 cv::IMREAD_COLOR);

  cv::Mat grayscale;
  cv::cvtColor(color_image, grayscale, cv::COLOR_BGR2GRAY);

  cv::Mat_<float> normalized;
  grayscale.convertTo(normalized, CV_32FC1, 1.0 / 255.0);
  const cv::Mat_<float> fake_image =
      (cv::Mat_<float>(5, 5) << 0, 50, 255, 50, 0, 0, 50, 255, 50, 0, 0, 50,
       255, 50, 0, 0, 50, 255, 50, 0, 0, 50, 255, 50, 0);
  cv::Mat_<float> fake_normalized;
  fake_image.convertTo(fake_normalized, CV_32FC1, 1.0 / 255.0);

  const std::vector<Kernel> edge_kernels = {SOBEL3x3, SOBEL5x5};
  const std::vector<Kernel> other_kernels;// = {GAUSSIAN3x3, GAUSSIAN5x5};
  std::vector<cv::Mat_<float>> edge_detected;
  std::vector<cv::Mat_<float>> other;
  for (const Kernel kernel : edge_kernels) {
    edge_detected.push_back(get_kerneled(normalized, kernel, true));
  }
  for (const Kernel kernel : other_kernels) {
    other.push_back(get_kerneled(normalized, kernel, false));
  }

  resize(edge_detected);
  resize(other);
  cv::Mat_<float> combined_edge;
  cv::hconcat(edge_detected, combined_edge);
  cv::imshow("Edge Kernels", combined_edge);
  cv::Mat_<float> combined_other;
  cv::hconcat(other, combined_other);
  cv::imshow("Other Kernels", combined_other);
  cv::waitKey(0);
  cv::destroyAllWindows();

  //
  // for (double threshold = 0.01; threshold < 0.3; threshold += 0.01) {
  //   cv::Mat_<float> cannied = edge_detection::Canny(fake_image, threshold);
  //   cv::imshow(std::format("Threshold: %f", threshold), cannied);
  //   cv::waitKey(0);
  // }

  return 0;
}