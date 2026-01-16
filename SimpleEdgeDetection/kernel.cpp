//
// Created by Yasen on 1/10/26.
//

#include "kernel.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat_<float> edge_detection::ApplyKernel(const cv::Mat_<float> &mat,
                                            const Kernel kernel_type,
                                            const int stride, const int padding,
                                            const bool transpose) {
  const cv::Mat_<float> &kernel =
      transpose ? kernels.at(kernel_type).t() : kernels.at(kernel_type);
  cv::Mat_<float> output = Convolve(mat, kernel, stride, padding);
  return output;
}

cv::Mat_<float> edge_detection::Convolve(const cv::Mat_<float> &mat,
                                         const cv::Mat_<float> &kernel,
                                         const int stride, const int padding) {
  const int kernel_sz = kernel.rows;
  const int crossCorrelatedRows =
      (mat.rows + 2 * padding - kernel_sz) / stride + 1;
  const int crossCorrelatedCols =
      (mat.cols + 2 * padding - kernel_sz) / stride + 1;
  cv::Mat_<float> output =
      cv::Mat_<float>::zeros(crossCorrelatedRows, crossCorrelatedCols);
  cv::Mat_<float> padded;
  cv::copyMakeBorder(mat, padded, padding, padding, padding, padding,
                     cv::BORDER_CONSTANT, cv::Scalar(0));
  for (int row = 0; row < crossCorrelatedRows; row++) {
    for (int col = 0; col < crossCorrelatedCols; col++) {
      output(row, col) = FrobeniusInner(
          padded(cv::Rect{col * stride, row * stride, kernel_sz, kernel_sz}),
          kernel);
    }
  }
  return output;
}

cv::Mat_<float> edge_detection::CleanupEdges(const cv::Mat_<float> &image,
                                             const float accept_threshold,
                                             const float consider_threshold) {
  cv::Mat_<float> thresholded(image.rows, image.cols);
  float average_edge_likelihood = 0;
  int num_edges = 0;
  for (int row = 0; row < image.rows; row++) {
    for (int col = 0; col < image.rows; col++) {
      if (image(row, col) >= accept_threshold) {
        thresholded(row, col) = 255;
        average_edge_likelihood += image(row, col);
        num_edges++;
      } else if (image(row, col) >= consider_threshold) {
        bool edge_adjacent = false;
        for (const std::pair<int, int> &dir : adjacent_edge_check_directions) {
          const int check_row = row + dir.first;
          const int check_col = col + dir.second;
          if (check_row < 0 || check_row >= thresholded.rows || check_col < 0 ||
              check_col >= thresholded.cols) {
            continue;
          }
          if (thresholded(check_row, check_col) == 255) {
            edge_adjacent = true;
            break;
          }
        }
        if (edge_adjacent) {
          thresholded(row, col) = 255;
          average_edge_likelihood += image(row, col);
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
