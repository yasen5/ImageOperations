//
// Created by Yasen on 1/10/26.
//

#ifndef IMAGEGRADIENTS_EDGE_DETECTOR_H
#define IMAGEGRADIENTS_EDGE_DETECTOR_H

#include <Eigen/Dense>
#include <map>
#include <opencv2/core/eigen.hpp>

using namespace Eigen;

using Image = Matrix<uint8_t, Dynamic, Dynamic>;

enum Kernel { ROBERTS, SOBEL3x3, SOBEL5x5, GAUSSIAN3x3, GAUSSIAN5x5, IDENTITY };

namespace edge_detection {
MatrixXf ApplyKernel(const cv::Mat &cv_img, Kernel kernel_type, int stride = 1,
                     int padding = 0);
MatrixXf ApplyKernel(const MatrixXf &mat, Kernel kernel_type, int stride = 1,
                     int padding = 0);
Image CleanupEdges(const MatrixXf &mat, float accept_threshold,
                   float consider_threshold);

MatrixXf Convolve(const MatrixXf &mat, const MatrixXf &kernel, int stride = 1,
                  int padding = 0);
inline float FrobeniusInner(const MatrixXf &operand, const MatrixXf &mat) {
  return (operand.array() * mat.array()).sum();
}
inline const std::map<Kernel, MatrixXf> kernels = {
    {ROBERTS, (Matrix2f() << 0, 1, -1, 0).finished()},
    {SOBEL3x3, (Matrix3f() << -1, 0, 1, -2, 0, 2, -1, 0, 1).finished()},
    {SOBEL5x5, (MatrixXf(5, 5) << 2, 2, 4, 2, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0,
                -1, -1, -2, -1, -1, -2, -2, -4, -2, -2)
                   .finished()},
    {GAUSSIAN3x3, (Matrix3f() << 1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 8.0,
                   1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0)
                      .finished()},
    {GAUSSIAN5x5, (MatrixXf(5, 5) << 1, 4, 7, 4, 1, 4, 16, 26, 16, 4, 7, 26, 41,
                   26, 7, 4, 16, 26, 16, 4, 1, 4, 7, 4, 1)
                          .finished() /
                      273.0},
    {IDENTITY, Matrix2f::Identity()},
};
constexpr std::array<std::pair<int, int>, 4> adjacent_edge_check_directions = {
    {{-1, -1}, {-1, 0}, {0, -1}, {-1, 1}}}; // row adjustment, column adjustment
}; // namespace edge_detection

#endif // IMAGEGRADIENTS_EDGE_DETECTOR_H