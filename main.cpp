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
  const std::string img_name = "test_images/test_image5.jpg";
  const cv::Mat color_image =
      cv::imread("/Users/yasen/CLionProjects/ImageGradients/" + img_name,
                 cv::IMREAD_COLOR);

  cv::Mat grayscale =
      cv::imread("/Users/yasen/CLionProjects/ImageGradients/" + img_name,
                 cv::IMREAD_GRAYSCALE);
  cv::cvtColor(color_image, grayscale, cv::COLOR_BGR2GRAY);

  cv::Mat_<float> normalized;
  grayscale.convertTo(normalized, CV_32FC1, 1.0 / 255.0);
  // const cv::Mat_<float> fake_image =
  //     (cv::Mat_<float>(5, 5) << 0, 50, 255, 50, 0, 0, 50, 255, 50, 0, 0, 50,
  //      255, 50, 0, 0, 50, 255, 50, 0, 0, 50, 255, 50, 0);
  // cv::Mat_<float> fake_normalized;
  // fake_image.convertTo(fake_normalized, CV_32FC1, 1.0 / 255.0);

  // clang-format off
  const cv::Mat_<float> wall_image =
      (cv::Mat_<float>(20, 24) <<
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0.8, 0.8, 0.8, 0.8, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        );
  // clang-format on
  wall_image.convertTo(wall_image, CV_32FC1, 1.0);

  cv::Mat cannied = edge_detection::Canny(normalized, false);
  cv::imshow(std::format("No hist"), cannied);
  cv::Mat hist = edge_detection::Canny(normalized, true);
  cv::imshow(std::format("Hist"), hist);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}