//
// Created by Yasen on 1/12/26.
//

#include "canny.h"

#include <iostream>
cv::Mat_<float> edge_detection::Canny(const cv::Mat_<float> &image,
                                      const float consideration_threshold) {
  cv::Mat_<float> blurred = Convolve(image, GaussianKernel(10, 4));
  cv::Mat_<float> edges_x = abs(ApplyKernel(image, SOBEL3x3, 1, 0, false));
  cv::Mat_<float> edges_y = abs(ApplyKernel(image, SOBEL3x3, 1, 0, true));
  cv::Mat_<float> total_edges = edges_x.mul(edges_x) + edges_y.mul(edges_y);
  const int search_distance = 3;
  for (int row = 0; row < total_edges.rows; row++) {
    for (int col = 0; col < total_edges.cols; col++) {
      float x_gradient = edges_x.at<float>(row, col);
      float y_gradient = edges_y.at<float>(row, col);
      if (std::hypot(x_gradient, y_gradient) < consideration_threshold)
        continue;
      const float scalar = std::max(x_gradient, y_gradient);
      x_gradient /= scalar;
      y_gradient /= scalar;
      float max_value = 0;
      int max_index = INFINITY;
      for (int i = -search_distance / 2; i < search_distance / 2; i++) {
        const int check_col = col + static_cast<int>(i * x_gradient);
        const int check_row = row + static_cast<int>(i * y_gradient);
        if (check_col < 0) {
          if (x_gradient < 0) {
            break;
          }
          continue;
        }
        if (check_row < 0) {
          if (y_gradient < 0) {
            break;
          }
          continue;
        }
        if (check_col >= total_edges.cols) {
          if (x_gradient > 0) {
            break;
          }
          continue;
        }
        if (check_row >= total_edges.rows) {
          if (y_gradient > 0) {
            break;
          }
          continue;
        }
        if (total_edges.at<float>(check_row, check_col) > max_value) {
          max_value = total_edges.at<float>(check_row, check_col);
          max_index = i;
        }
      }
      for (int i = -search_distance / 2; i < search_distance / 2; i++) {
        constexpr int thickness = 0;
        const int check_col = col + static_cast<int>(i * x_gradient);
        const int check_row = row + static_cast<int>(i * y_gradient);
        if (check_col < 0) {
          if (x_gradient < 0) {
            break;
          }
          continue;
        }
        if (check_row < 0) {
          if (y_gradient < 0) {
            break;
          }
          continue;
        }
        if (check_col >= total_edges.cols) {
          if (x_gradient > 0) {
            break;
          }
          continue;
        }
        if (check_row >= total_edges.rows) {
          if (y_gradient > 0) {
            break;
          }
          continue;
        }
        total_edges.at<float>(check_row, check_col) =
            (std::abs(i - max_index) <= thickness / 2) ? 1 : 0;
      }
    }
  }
  return total_edges;
}