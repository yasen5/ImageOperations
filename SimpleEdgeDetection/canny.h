//
// Created by Yasen on 1/12/26.
//

#ifndef IMAGEGRADIENTS_CANNY_H
#define IMAGEGRADIENTS_CANNY_H
#include "kernel.h"

#include <Eigen/src/Core/Matrix.h>
#include <opencv2/core/mat.hpp>

namespace edge_detection {
constexpr float NMS_THRESHOLD = 0.06;
constexpr float GAUSSIAN_SIGMA = 1.8;
constexpr float HISTERESIS_RIDGE_START_THRESHOLD = 0.3;
constexpr float HISTERESIS_RIDGE_CONTINUE_THRESHOLD = NMS_THRESHOLD;
constexpr int HISTERESIS_CLIFF_LENGTH = 5;
constexpr int THICKNESS = 2;
constexpr int ALLOWED_CANNY_GAP = 5;
constexpr float EDGE_ANGLE_DISPARITY_TOLERANCE = 20 * M_PI / 180;
cv::Mat Canny(const cv::Mat_<float> &image, int stride, bool histeresis,
              float nms_threshold = NMS_THRESHOLD,
              float gaussian_sigma = GAUSSIAN_SIGMA);
void ThinEdges(cv::Mat &total_edges, cv::Mat &x_edges, cv::Mat &y_edges,
               float nms_threshold);
cv::Mat_<float> Histeresis(
    cv::Mat_<float> &x_edges, cv::Mat_<float> &y_edges,
    float ridge_start_threshold = HISTERESIS_RIDGE_START_THRESHOLD,
    float ridge_continue_threshold = HISTERESIS_RIDGE_CONTINUE_THRESHOLD);
cv::Mat_<float> Binary(const cv::Mat_<float> &image, float lower_threshold,
                       float upper_threshold);
void PerpendicularSlope(float &dx, float &dy);
void CullTheWeak(cv::Mat_<float> &edges);
cv::Mat ColorEdges(const cv::Mat_<float> &total_edges,
                   const cv::Mat_<float> &x_edges,
                   const cv::Mat_<float> &y_edges);
inline float ContainedAngle(float x1, float y1, float x2, float y2) {
  return std::asin((x1 * y2 - x2 * y1) /
                   (std::hypot(x1, y1) * std::hypot(x2, y2)));
}
} // namespace edge_detection

#endif // IMAGEGRADIENTS_CANNY_H
