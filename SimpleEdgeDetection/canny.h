//
// Created by Yasen on 1/12/26.
//

#ifndef IMAGEGRADIENTS_CANNY_H
#define IMAGEGRADIENTS_CANNY_H
#include "kernel.h"

#include <Eigen/src/Core/Matrix.h>
#include <opencv2/core/mat.hpp>

namespace edge_detection {
cv::Mat Canny(const cv::Mat &image, float consideration_threshold);
} // namespace edge_detection

#endif // IMAGEGRADIENTS_CANNY_H
