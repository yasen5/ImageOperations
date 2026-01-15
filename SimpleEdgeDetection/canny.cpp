//
// Created by Yasen on 1/12/26.
//

#include "canny.h"

#include <iostream>
cv::Mat edge_detection::Canny(const cv::Mat &image,
                              const float consideration_threshold) {
  MatrixXf blurred = ApplyKernel(image, GAUSSIAN3x3);
  MatrixXf edges_x = ApplyKernel(image, SOBEL3x3).cwiseAbs();
  MatrixXf edges_y = ApplyKernel(image, SOBEL3x3).cwiseAbs();
  MatrixXf total_edges = edges_x + edges_y;
  // std::cout << "Image average: " << total_edges.sum() / total_edges.size()
  //           << std::endl;
  const int search_distance = image.rows * image.cols / 10000;
  // std::cout << "Search distance: " << search_distance << std::endl;
  size_t num_deletions = 0;
  std::cout << "Threshold: " << consideration_threshold << std::endl;
  for (size_t row = 0; row < total_edges.rows(); row++) {
    for (size_t col = 0; col < total_edges.cols(); col++) {
      float x_slope = edges_x(row, col);
      float y_slope = edges_y(row, col);
      if (std::hypot(x_slope, y_slope) < consideration_threshold)
        continue;
      const float scalar = std::max(x_slope, y_slope);
      x_slope /= scalar;
      y_slope /= scalar;
      float max_value = 0;
      for (int i = -search_distance / 2; i < search_distance / 2; i++) {
        const int check_col = col + static_cast<size_t>(i * x_slope);
        const int check_row = row + static_cast<size_t>(i * y_slope);
        if (check_col < 0) {
          if (x_slope < 0) {
            break;
          }
          continue;
        }
        if (check_row < 0) {
          if (y_slope < 0) {
            break;
          }
          continue;
        }
        if (check_col >= total_edges.cols()) {
          if (x_slope > 0) {
            break;
          }
          continue;
        }
        if (check_row >= total_edges.rows()) {
          if (y_slope > 0) {
            break;
          }
          continue;
        }
        if (total_edges(check_row, check_col) > max_value) {
          max_value = total_edges(check_row, check_col);
        }
      }
      for (int i = -search_distance / 2; i < search_distance / 2; i++) {
        const int check_col = col + static_cast<size_t>(i * x_slope);
        const int check_row = row + static_cast<size_t>(i * y_slope);
        if (check_col < 0) {
          if (x_slope < 0) {
            break;
          }
          continue;
        }
        if (check_row < 0) {
          if (y_slope < 0) {
            break;
          }
          continue;
        }
        if (check_col >= total_edges.cols()) {
          if (x_slope > 0) {
            break;
          }
          continue;
        }
        if (check_row >= total_edges.rows()) {
          if (y_slope > 0) {
            break;
          }
          continue;
        }
        if (total_edges(check_row, check_col) != max_value) {
          total_edges(check_row, check_col) = 0;
          num_deletions++;
        }
      }
    }
  }
  if (num_deletions == 0) {
    std::cout << "Deleted nothing" << std::endl;
  }
  Matrix<float32_t, Dynamic, Dynamic, RowMajor> row_major = total_edges;
  cv::Mat cvmat(row_major.rows(), row_major.cols(), CV_32FC1, row_major.data());
  return cvmat;
}