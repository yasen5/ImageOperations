//
// Created by Yasen on 1/12/26.
//

#include "canny.h"

#include <iostream>
cv::Mat_<float> edge_detection::Canny(const cv::Mat_<float> &image,
                                      const float nms_threshold,
                                      const float sigma,
                                      const int gaussian_kernel_size,
                                      const int search_distance) {
  const cv::Mat_<float> blurred =
      Convolve(image, GaussianKernel(gaussian_kernel_size, sigma));
  cv::Mat_<float> edges_x = abs(ApplyKernel(blurred, SOBEL3x3, 1, 0, false));
  cv::Mat_<float> edges_y = abs(ApplyKernel(blurred, SOBEL3x3, 1, 0, true));
  Histeresis(edges_x, edges_y);
  cv::Mat_<float> total_edges = edges_x.mul(edges_x) + edges_y.mul(edges_y);
  sqrt(total_edges, total_edges);
  for (int row = 0; row < total_edges.rows; row++) {
    for (int col = 0; col < total_edges.cols; col++) {
      float x_gradient = edges_x.at<float>(row, col);
      float y_gradient = edges_y.at<float>(row, col);
      if (std::hypot(x_gradient, y_gradient) < nms_threshold)
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

cv::Mat_<float> edge_detection::Histeresis(cv::Mat_<float> &x_edges,
                                           cv::Mat_<float> &y_edges,
                                           float ridge_start_threshold,
                                           float ridge_continue_threshold) {
  cv::Mat_<float> edge_strength_map(x_edges.rows, x_edges.cols);
  std::vector<cv::Point2i> starting_points;
  for (int row = 0; row < x_edges.rows; row++) {
    for (int col = 0; col < x_edges.cols; col++) {
      float edge_strength =
          std::hypot(x_edges.at<float>(row, col), y_edges.at<float>(row, col));
      if (edge_strength > ridge_start_threshold) {
        starting_points.push_back({col, row});
      }
      edge_strength_map(row, col) = edge_strength;
    }
  }
  for (auto &point : starting_points) {
    float row = point.y;
    float col = point.x;
    float greatest_strength =
        edge_strength_map.at<float>(point); // greatest strength on this line
    for (int sign = -1; sign < 2; sign += 2) {
      while (true) {
        float x_slope = x_edges.at<float>(row, col);
        float y_slope = y_edges.at<float>(row, col);
        float scalar = std::max(x_slope, y_slope);
        x_slope /= scalar;
        y_slope /= scalar;
        PerpendicularSlope(x_slope, y_slope);
        row += y_slope * sign;
        col += x_slope * sign;
        if (row < 0 || row >= x_edges.rows || col < 0 || col >= x_edges.cols) {
          row -= y_slope * sign;
          col -= x_slope * sign;
          break;
        }
        float edge_strength = edge_strength_map.at<float>(point);
        if (edge_strength < ridge_continue_threshold) {
          break;
        }
        if (edge_strength > greatest_strength) {
          greatest_strength = edge_strength;
        }
      }
      while (std::abs(col - point.x) < 0.01 && std::abs(row - point.y) < 0.01) {
        edge_strength_map.at<float>(row, col) = greatest_strength;
        float x_slope = x_edges.at<float>(row, col);
        float y_slope = y_edges.at<float>(row, col);
        float scalar = std::max(x_slope, y_slope);
        x_slope /= scalar;
        y_slope /= scalar;
        PerpendicularSlope(x_slope, y_slope);
        x_slope *= -1;
        y_slope *= -1;
        float strength_scalar =
            greatest_strength / edge_strength_map.at<float>(row, col);
        edge_strength_map.at<float>(row, col) = greatest_strength;
        x_edges.at<float>(row, col) *= strength_scalar;
        y_edges.at<float>(row, col) *= strength_scalar;
        col += x_slope;
        row += y_slope;
      }
    }
  }
}

void edge_detection::PerpendicularSlope(float &dx, float &dy) {
  float temp = dx;
  dx = -dy;
  dy = temp;
}
