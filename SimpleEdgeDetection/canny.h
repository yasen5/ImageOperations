//
// Created by Yasen on 1/12/26.
//

#ifndef IMAGEGRADIENTS_CANNY_H
#define IMAGEGRADIENTS_CANNY_H
#include "kernel.h"

#include <Eigen/src/Core/Matrix.h>
#include <opencv2/core/mat.hpp>

namespace edge_detection {
constexpr float NMS_THRESHOLD = 0.08;
constexpr float GAUSSIAN_SIGMA = 1.8;
constexpr int GAUSSIAN_KERNEL_SIZE = 5.0;
constexpr int SEARCH_DISTANCE = 5;
constexpr float HISTERESIS_RIDGE_START_THRESHOLD = 0.16;
constexpr float HISTERESIS_RIDGE_CONTINUE_THRESHOLD = NMS_THRESHOLD;
constexpr int HISTERESIS_CLIFF_LENGTH = 7;
constexpr int THICKNESS = 2;
constexpr int ALLOWED_CANNY_GAP = 5;
cv::Mat Canny(const cv::Mat_<float> &image, bool histeresis,
              float nms_threshold = NMS_THRESHOLD,
              float gaussian_sigma = GAUSSIAN_SIGMA,
              int gaussian_kernel_size = GAUSSIAN_KERNEL_SIZE,
              int search_distance = SEARCH_DISTANCE);
void ThinEdges(cv::Mat &total_edges, cv::Mat &x_edges, cv::Mat &y_edges);
cv::Mat_<float> Histeresis(
    cv::Mat_<float> &x_edges, cv::Mat_<float> &y_edges,
    float ride_start_threshold = HISTERESIS_RIDGE_START_THRESHOLD,
    float ridge_continue_threshold = HISTERESIS_RIDGE_CONTINUE_THRESHOLD);
cv::Mat_<float> Binary(const cv::Mat_<float> &image, float lower_threshold,
                       float upper_threshold);
void PerpendicularSlope(float &dx, float &dy);
void CullTheWeak(cv::Mat_<float> &edges);
cv::Mat ColorEdges(const cv::Mat_<float> &total_edges,
                   const cv::Mat_<float> &x_edges,
                   const cv::Mat_<float> &y_edges);
} // namespace edge_detection

#endif // IMAGEGRADIENTS_CANNY_H
