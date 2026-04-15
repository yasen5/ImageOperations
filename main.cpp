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
  cv::Mat grayscale = cv::imread("/Users/yasen/Downloads/"
                                 "YellowBall.jpg",
                                 cv::IMREAD_GRAYSCALE);
  // cv::Mat grayscale =
  //     cv::imread("/Users/yasen/Documents/Wallpapers/"
  //                "Surgehacker-Mech-Kamigawa-Neon-Dynasty-MtG-Art.jpg",
  //                cv::IMREAD_GRAYSCALE);

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

  cv::Mat cannied = edge_detection::Canny(normalized, 5, false);
  cv::imshow(std::format("No hist"), cannied);
  cv::Mat hist = edge_detection::Canny(normalized, 5, true);
  cv::imshow(std::format("Hist"), hist);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}