//
// Created by Yasen on 1/10/26.
//

#include "edge_detector.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace Eigen;

MatrixXf edge_detection::ApplyKernel(const cv::Mat &cv_img,
                                     const Kernel kernel_type, const int stride,
                                     const int padding) {
  cv::Mat normalized;
  if (cv_img.channels() == 3) {
    cv::cvtColor(cv_img, normalized, cv::COLOR_BGR2GRAY);
  } else {
    normalized = cv_img;
  }
  normalized.convertTo(normalized, CV_32FC1, 1.0 / 255.0);
  const Map<Matrix<float32_t, Dynamic, Dynamic, RowMajor>> eigen_mat(
      normalized.ptr<float32_t>(), normalized.rows, normalized.cols);
  return ApplyKernel(eigen_mat, kernel_type, stride, padding);
}

MatrixXf edge_detection::ApplyKernel(const MatrixXf &mat,
                                     const Kernel kernel_type, const int stride,
                                     const int padding) {
  const MatrixXf &kernel = kernels.at(kernel_type);
  const bool inverted_symmetric =
      kernel.col(0).isApprox(-kernel.col(kernel.cols() - 1)) ||
      kernel.row(0).isApprox(-kernel.row(kernel.rows() - 1));
  MatrixXf output = Convolve(mat, kernel, stride, padding);
  if (inverted_symmetric) {
    output += Convolve(mat, kernel.transpose(), stride, padding);
    output /= 2;
  }
  // std::cout << "MaxCoeff: " << output.maxCoeff() << std::endl;
  // output /= output.maxCoeff();
  return output;
}

MatrixXf edge_detection::Convolve(const MatrixXf &mat, const MatrixXf &kernel,
                                  const int stride, const int padding) {
  const int kernel_sz = kernel.rows();
  const int crossCorrelatedRows =
      (mat.rows() + 2 * padding - kernel_sz) / stride + 1;
  const int crossCorrelatedCols =
      (mat.cols() + 2 * padding - kernel_sz) / stride + 1;
  MatrixXf padded =
      MatrixXf::Zero(mat.rows() + 2 * padding, mat.cols() + 2 * padding);
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

Image edge_detection::CleanupEdges(const MatrixXf &mat,
                                   const float accept_threshold,
                                   const float consider_threshold) {
  Image thresholded =
      Matrix<uint8_t, Dynamic, Dynamic>::Zero(mat.rows(), mat.cols());
  float average_edge_likelihood = 0;
  int num_edges = 0;
  for (int row = 0; row < mat.rows(); row++) {
    for (int col = 0; col < mat.rows(); col++) {
      if (mat(row, col) >= accept_threshold) {
        thresholded(row, col) = 255;
        average_edge_likelihood += mat(row, col);
        num_edges++;
      } else if (mat(row, col) >= consider_threshold) {
        bool edge_adjacent = false;
        for (const std::pair<int, int> &dir : adjacent_edge_check_directions) {
          const int check_row = row + dir.first;
          const int check_col = col + dir.second;
          if (check_row < 0 || check_row >= thresholded.rows() ||
              check_col < 0 || check_col >= thresholded.cols()) {
            continue;
          }
          if (thresholded(check_row, check_col) == 255) {
            edge_adjacent = true;
            break;
          }
        }
        if (edge_adjacent) {
          thresholded(row, col) = 255;
          average_edge_likelihood += mat(row, col);
          num_edges++;
        }
      }
    }
  }
  average_edge_likelihood /= num_edges;
  std::cout << "Average edge likelihood: " << average_edge_likelihood
            << std::endl;
  return thresholded;
}
