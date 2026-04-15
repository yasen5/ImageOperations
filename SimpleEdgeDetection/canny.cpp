//
// Created by Yasen on 1/12/26.
//

#include "canny.h"

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using weighted_index_t = struct WeightedIndex {
  int row;
  int col;
  float weight;

  bool operator<(const WeightedIndex &right) const {
    return weight < right.weight;
  }
};

cv::Mat edge_detection::Canny(const cv::Mat_<float> &image, int stride,
                              const bool histeresis, const float nms_threshold,
                              const float sigma) {
  const cv::Mat_<float> blurred =
      Convolve(image, GaussianKernel(stride, sigma), stride, 0);
  cv::Mat_<float> x_edges = ApplyKernel(blurred, SOBEL3x3, 1, 0, false);
  cv::Mat_<float> y_edges = ApplyKernel(blurred, SOBEL3x3, 1, 0, true);
  cv::Mat_<float> total_edges = x_edges.mul(x_edges) + y_edges.mul(y_edges);
  sqrt(total_edges, total_edges);
  ThinEdges(total_edges, x_edges, y_edges, nms_threshold);
  if (histeresis) {
    total_edges = Histeresis(x_edges, y_edges);
    ThinEdges(total_edges, x_edges, y_edges, nms_threshold);
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
      size_t num_below_continue_threshold = 0;
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
        const float edge_strength = edge_strength_map.at<float>(row, col);
        if (edge_strength >= starting_edge_strength) {
          break;
        }
        if (edge_strength < ridge_continue_threshold) {
          num_below_continue_threshold++;
          if (num_below_continue_threshold >= HISTERESIS_CLIFF_LENGTH) {
            break;
          }
        } else {
          num_below_continue_threshold = 0;
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
      edges.at<float>(row, col) =
          (edges.at<float>(row, col) < NMS_THRESHOLD) ? 0 : 1;
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
        if (edges.at<float>(new_row, new_col) > NMS_THRESHOLD) {
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

cv::Mat edge_detection::ColorEdges(const cv::Mat_<float> &total_edges,
                                   const cv::Mat_<float> &x_edges,
                                   const cv::Mat_<float> &y_edges) {
  cv::Mat angle, normalized, color;

  cv::phase(x_edges, y_edges, angle, true);

  angle.convertTo(normalized, CV_8U, 255.0 / 360.0);

  // Apply colormap (expects CV_8UC1)
  cv::applyColorMap(normalized, color, cv::COLORMAP_RAINBOW);

  for (int row = 0; row < color.rows; row++) {
    for (int col = 0; col < color.cols; col++) {
      if (total_edges(row, col) < HISTERESIS_RIDGE_START_THRESHOLD) {
        color.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 0);
      }
    }
  }

  return color;
}

void edge_detection::ThinEdges(cv::Mat &total_edges, cv::Mat &x_edges,
                               cv::Mat &y_edges, const float nms_threshold) {
  std::priority_queue<weighted_index_t> starting_points;
  for (int row = 0; row < total_edges.rows; row++) {
    for (int col = 0; col < total_edges.cols; col++) {
      const float edge_strength =
          std::hypot(x_edges.at<float>(row, col), y_edges.at<float>(row, col));
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
    cv::Point2f slope = {x_edges.at<float>(row, col),
                         y_edges.at<float>(row, col)};
    slope /= std::max(std::abs(slope.x), std::abs(slope.y));
    for (int sign = -1; sign < 2; sign += 2) {
      int i = 0;
      int num_blank = 0;
      while (true) {
        i++;
        const int check_col = col + sign * lround(i * slope.x);
        const int check_row = row + sign * lround(i * slope.y);
        if (check_col < 0) {
          if (sign * slope.x < 0) {
            break;
          }
          continue;
        }
        if (check_row < 0) {
          if (sign * slope.y < 0) {
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
          if (sign * slope.y > 0) {
            break;
          }
          continue;
        }
        if (check_col == col && check_row == row) {
          continue;
        }
        if (total_edges.at<float>(check_row, check_col) <
            HISTERESIS_RIDGE_CONTINUE_THRESHOLD) {
          num_blank++;
          if (num_blank > ALLOWED_CANNY_GAP) {
            break;
          }
        }
        if (i > THICKNESS / 2) {
          total_edges.at<float>(check_row, check_col) = 0;
          x_edges.at<float>(check_row, check_col) = 0;
          y_edges.at<float>(check_row, check_col) = 0;
        }
      }
    }
  }
}
