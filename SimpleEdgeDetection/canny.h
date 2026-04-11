//
// Created by Yasen on 1/12/26.
//

#ifndef IMAGEGRADIENTS_CANNY_H
#define IMAGEGRADIENTS_CANNY_H
#include "kernel.h"

#include <Eigen/src/Core/Matrix.h>
#include <opencv2/core/mat.hpp>

namespace edge_detection {
constexpr float NMS_THRESHOLD = 0.04;
constexpr float GAUSSIAN_SIGMA = 1.8;
constexpr int GAUSSIAN_KERNEL_SIZE = 5.0;
constexpr int SEARCH_DISTANCE = 20;
constexpr float HISTERESIS_RIDGE_START_THRESHOLD = 0.16;
constexpr float HISTERESIS_RIDGE_CONTINUE_THRESHOLD = 0.1;
cv::Mat_<float> Canny(const cv::Mat_<float> &image, bool histeresis,
                      float nms_threshold = NMS_THRESHOLD,
                      float gaussian_sigma = GAUSSIAN_SIGMA,
                      int gaussian_kernel_size = GAUSSIAN_KERNEL_SIZE,
                      int search_distance = SEARCH_DISTANCE);
cv::Mat_<float> Histeresis(
    cv::Mat_<float> &x_edges, cv::Mat_<float> &y_edges,
    float ride_start_threshold = HISTERESIS_RIDGE_START_THRESHOLD,
    float ridge_continue_threshold = HISTERESIS_RIDGE_CONTINUE_THRESHOLD);
cv::Mat_<float> Binary(const cv::Mat_<float> &image, float lower_threshold,
                       float upper_threshold);
void PerpendicularSlope(float &dx, float &dy);
void CullTheWeak(cv::Mat_<float> &edges);
} // namespace edge_detection

#endif // IMAGEGRADIENTS_CANNY_H
