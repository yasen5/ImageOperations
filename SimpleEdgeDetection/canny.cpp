//
// Created by Yasen on 1/12/26.
//

#include "canny.h"

#include <iostream>
cv::Mat edge_detection::Canny(const cv::Mat &image,
                              const float consideration_threshold) {
  // MatrixXf blurred = ApplyKernel(image, GAUSSIAN3x3);
  MatrixXf edges_x = ApplyKernel(image, SOBEL3x3).cwiseAbs();
  MatrixXf edges_y = ApplyKernel(image, SOBEL3x3, true).cwiseAbs();
  MatrixXf total_edges =
      (edges_x.cwiseSquare() + edges_y.cwiseSquare()).cwiseSqrt();
  const int search_distance = total_edges.rows() * total_edges.cols() / 1000;
  for (size_t row = 0; row < total_edges.rows(); row++) {
    for (size_t col = 0; col < total_edges.cols(); col++) {
      float x_gradient = edges_x(row, col);
      float y_gradient = edges_y(row, col);
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
        if (check_col >= total_edges.cols()) {
          if (x_gradient > 0) {
            break;
          }
          continue;
        }
        if (check_row >= total_edges.rows()) {
          if (y_gradient > 0) {
            break;
          }
          continue;
        }
        if (total_edges(check_row, check_col) > max_value) {
          max_value = total_edges(check_row, check_col);
          max_index = i;
        }
      }
      for (int i = -search_distance / 2; i < search_distance / 2; i++) {
        constexpr int thickness = 50;
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
        if (check_col >= total_edges.cols()) {
          if (x_gradient > 0) {
            break;
          }
          continue;
        }
        if (check_row >= total_edges.rows()) {
          if (y_gradient > 0) {
            break;
          }
          continue;
        }
        total_edges(check_row, check_col) =
            (std::abs(i - max_index) <= thickness / 2) ? 1 : 0;
      }
    }
  }
  Matrix<float32_t, Dynamic, Dynamic, RowMajor> row_major = total_edges;
  cv::Mat cvmat(row_major.rows(), row_major.cols(), CV_32FC1, row_major.data());
  return cvmat;
}