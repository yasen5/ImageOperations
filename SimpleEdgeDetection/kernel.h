//
// Created by Yasen on 1/10/26.
//

#ifndef IMAGEGRADIENTS_EDGE_DETECTOR_H
#define IMAGEGRADIENTS_EDGE_DETECTOR_H

#include <Eigen/Dense>
#include <map>
#include <opencv2/core/eigen.hpp>

using namespace Eigen;

enum Kernel { ROBERTS, SOBEL3x3, SOBEL5x5, GAUSSIAN3x3, GAUSSIAN5x5 };

namespace edge_detection {
cv::Mat_<float> ApplyKernel(const cv::Mat_<float> &mat, Kernel kernel_type, int stride = 1,
                 int padding = 0, bool transpose = false);
cv::Mat_<float> CleanupEdges(const cv::Mat_<float> &image, float accept_threshold,
                  float consider_threshold);
cv::Mat_<float> Convolve(const cv::Mat_<float> &mat, const cv::Mat_<float> &kernel, int stride = 1,
              int padding = 0);
inline float FrobeniusInner(const cv::Mat_<float> &a, const cv::Mat_<float> &b) { return a.dot(b); }
inline const std::map<Kernel, cv::Mat_<float>> kernels = {
    {ROBERTS, (cv::Mat_<float>(2, 2) << 0, 1, -1, 0)},
    {SOBEL3x3, (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1)},
    {SOBEL5x5, (cv::Mat_<float>(5, 5) << 2, 2, 4, 2, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, -1,
                -1, -2, -1, -1, -2, -2, -4, -2, -2)},
    {GAUSSIAN3x3, (cv::Mat_<float>(3, 3) << 1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 8.0,
                   1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0)},
    {GAUSSIAN5x5, (cv::Mat_<float>(5, 5) << 1, 4, 7, 4, 1, 4, 16, 26, 16, 4, 7, 26, 41, 26,
                   7, 4, 16, 26, 16, 4, 1, 4, 7, 4, 1) /
                      273.0},
};
constexpr std::array<std::pair<int, int>, 4> adjacent_edge_check_directions = {
    {{-1, -1}, {-1, 0}, {0, -1}, {-1, 1}}}; // row adjustment, column adjustment
}; // namespace edge_detection

#endif // IMAGEGRADIENTS_EDGE_DETECTOR_H