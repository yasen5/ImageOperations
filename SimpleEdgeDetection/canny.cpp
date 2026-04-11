//
// Created by Yasen on 1/12/26.
//

#include "canny.h"

#include <iostream>

using weighted_index_t = struct WeightedIndex {
  int row;
  int col;
  float weight;

  bool operator<(const WeightedIndex &right) const {
    return weight < right.weight;
  }
};

cv::Mat_<float> edge_detection::Canny(const cv::Mat_<float> &image,
                                      bool histeresis,
                                      const float nms_threshold,
                                      const float sigma,
                                      const int gaussian_kernel_size,
                                      const int search_distance) {
  const cv::Mat_<float> blurred =
      Convolve(image, GaussianKernel(gaussian_kernel_size, sigma));
  cv::Mat_<float> edges_x = abs(ApplyKernel(blurred, SOBEL3x3, 1, 0, false));
  cv::Mat_<float> edges_y = abs(ApplyKernel(blurred, SOBEL3x3, 1, 0, true));
  cv::Mat_<float> total_edges;
  if (histeresis) {
    total_edges = Histeresis(edges_x, edges_y);
    // return total_edges;
  } else {
    total_edges = edges_x.mul(edges_x) + edges_y.mul(edges_y);
    sqrt(total_edges, total_edges);
  }
  std::priority_queue<weighted_index_t> starting_points;
  for (int row = 0; row < total_edges.rows; row++) {
    for (int col = 0; col < total_edges.cols; col++) {
      const float edge_strength =
          std::hypot(edges_x.at<float>(row, col), edges_y.at<float>(row, col));
      if (edge_strength > nms_threshold) {
        starting_points.push({row, col, edge_strength});
      }
    }
  }
  while (!starting_points.empty()) {
    auto [row, col, strength] = starting_points.top();
    starting_points.pop();
    if (total_edges.at<float>(row, col) < 0.01) {
      continue;
    }
    cv::Point2f slope = {edges_x.at<float>(row, col),
                         edges_y.at<float>(row, col)};
    slope /= std::max(slope.x, slope.y);
    for (int sign = -1; sign < 2; sign += 2) {
      for (int i = 1; i <= search_distance; i++) {
        constexpr int thickness = 1;
        const int check_col = col + sign * lround(i * slope.x);
        const int check_row = row + sign * lround(i * slope.y);
        if (check_col < 0) {
          if (sign * slope.x < 0) {
            break;
          }
          continue;
        }
        if (check_row < 0) {
          if (sign * slope.x < 0) {
            break;
          }
          continue;
        }
        if (check_col >= total_edges.cols) {
          if (sign * slope.x > 0) {
            break;
          }
          continue;
        }
        if (check_row >= total_edges.rows) {
          if (sign * slope.x > 0) {
            break;
          }
          continue;
        }
        if (check_col == col && check_row == row) {
          continue;
        }
        total_edges.at<float>(check_row, check_col) =
            (i <= thickness / 2) ? 1 : 0;
        // std::cout << "Gradx: " << slope.x << " grady: " << slope.y
        //           << " check_row: " << check_row << " check_col: " <<
        //           check_col
        //           << " row: " << row << " col: " << col << std::endl;
        // std::cout << "===" << std::endl;
      }
    }
  }
  // CullTheWeak(total_edges);
  return total_edges;
}

cv::Mat_<float> edge_detection::Histeresis(cv::Mat_<float> &x_edges,
                                           cv::Mat_<float> &y_edges,
                                           float ridge_start_threshold,
                                           float ridge_continue_threshold) {
  cv::Mat_<float> edge_strength_map =
      x_edges.mul(x_edges) + y_edges.mul(y_edges);
  sqrt(edge_strength_map, edge_strength_map);
  std::priority_queue<weighted_index_t> starting_points;
  for (int row = 0; row < x_edges.rows; row++) {
    for (int col = 0; col < x_edges.cols; col++) {
      if (edge_strength_map.at<float>(row, col) > ridge_start_threshold) {
        starting_points.push({row, col, edge_strength_map.at<float>(row, col)});
      }
    }
  }
  while (!starting_points.empty()) {
    auto [starting_row, starting_col, starting_edge_strength] =
        starting_points.top();
    starting_points.pop();
    for (int sign = -1; sign < 2; sign += 2) {
      int steps = 0;
      int row = starting_row, col = starting_col;
      while (true) {
        steps++;
        float x_slope = x_edges.at<float>(row, col);
        float y_slope = y_edges.at<float>(row, col);
        const float scalar = std::max(std::abs(x_slope), std::abs(y_slope));
        x_slope /= scalar;
        y_slope /= scalar;
        PerpendicularSlope(x_slope, y_slope);
        row = starting_row + lround(sign * y_slope * steps);
        col = starting_col + lround(sign * x_slope * steps);
        if (row == starting_row && col == starting_col) {
          continue;
        }
        if (row < 0 || row >= x_edges.rows || col < 0 || col >= x_edges.cols) {
          break;
        }
        if (const float edge_strength = edge_strength_map.at<float>(row, col);
            edge_strength < ridge_continue_threshold ||
            edge_strength >= starting_edge_strength) {
          break;
        }
        const float strength_scalar =
            starting_edge_strength / edge_strength_map.at<float>(row, col);
        edge_strength_map.at<float>(row, col) = starting_edge_strength;
        x_edges.at<float>(row, col) *= strength_scalar;
        y_edges.at<float>(row, col) *= strength_scalar;
      }
    }
  }
  return edge_strength_map;
}

void edge_detection::PerpendicularSlope(float &dx, float &dy) {
  const float temp = dx;
  dx = -dy;
  dy = temp;
}

void edge_detection::CullTheWeak(cv::Mat_<float> &edges) {
  for (size_t row = 0; row < edges.rows; row++) {
    for (size_t col = 0; col < edges.cols; col++) {
      if (edges.at<float>(row, col) < NMS_THRESHOLD) {
        edges.at<float>(row, col) = 0;
      }
    }
  }
  for (size_t row = 0; row < edges.rows; row++) {
    for (size_t col = 0; col < edges.cols; col++) {
      bool has_neighbor = false;
      for (const std::pair<int, int> &direction :
           adjacent_edge_check_directions_diagonal) {
        int new_row = row + direction.first;
        int new_col = col + direction.second;
        if (new_col < 0 || new_col >= edges.cols || new_row < 0 ||
            new_row >= edges.rows) {
          continue;
        }
        if (edges.at<float>(new_row, new_col) > 0) {
          has_neighbor = true;
          break;
        }
      }
      if (!has_neighbor) {
        edges.at<float>(row, col) = 0;
      }
    }
  }
}
