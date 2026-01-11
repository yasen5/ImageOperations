//
// Created by Yasen on 1/10/26.
//

#include "edge_detector.h"

#include <iostream>

using namespace Eigen;

MatrixXf EdgeDetector::ApplyKernel(const cv::Mat& cv_img, const Kernel kernel_type, const int stride, const int padding) {
    MatrixXf img;
    cv::cv2eigen<float_t>(cv_img, img);
    return ApplyKernel(img, kernel_type, stride, padding);
}

MatrixXf EdgeDetector::ApplyKernel(const MatrixXf &mat, const Kernel kernel_type, const int stride, const int padding) {
    const MatrixXf& kernel = kernels.at(kernel_type);
    // MatrixXf output = Convolve(mat, kernel, stride, padding) + Convolve(mat, kernel.transpose(), stride, padding);
    // output /= 2;
    return mat;
}

MatrixXf EdgeDetector::Convolve(const MatrixXf &mat, const MatrixXf &kernel, const int stride, const int padding) {
    const int kernel_sz = kernel.rows();
    const int crossCorrelatedRows =
        (mat.rows() + 2 * padding - kernel_sz) / stride + 1;
    const int crossCorrelatedCols =
        (mat.cols() + 2 * padding - kernel_sz) / stride + 1;
    MatrixXf padded = MatrixXf::Zero(mat.rows() + 2 * padding, mat.cols() + 2 * padding);
    MatrixXf output = MatrixXf::Zero(crossCorrelatedRows, crossCorrelatedCols);
    padded.block(padding, padding, mat.rows(), mat.cols()) = mat;
    for (int row = 0; row < crossCorrelatedRows; row++) {
        for (int col = 0; col < crossCorrelatedCols; col++) {
            output(row, col) = FrobeniusInner(
                padded.block(row * stride, col * stride, kernel_sz, kernel_sz),
                kernel);
        }
    }
    return output; 
}

