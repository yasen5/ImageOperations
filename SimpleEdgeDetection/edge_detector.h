//
// Created by Yasen on 1/10/26.
//

#ifndef IMAGEGRADIENTS_EDGE_DETECTOR_H
#define IMAGEGRADIENTS_EDGE_DETECTOR_H

#include <map>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

using namespace Eigen;

enum Kernel {
    ROBERTS,
    SOBEL,
    GAUSSIAN
};

class EdgeDetector {
public:
    static MatrixXf ApplyKernel(const cv::Mat& cv_img, Kernel kernel_type, int stride = 1, int padding = 1);
    static MatrixXf ApplyKernel(const MatrixXf& mat, Kernel kernel_type, int stride = 1, int padding = 1);
private:
    static MatrixXf Convolve(const MatrixXf& mat, const MatrixXf& kernel, int stride = 1, int padding = 1);
    static float FrobeniusInner(const MatrixXf& operand,
                                     const MatrixXf& mat) {
        return (operand.array() * mat.array()).sum();
    }
    inline static const std::map<Kernel, MatrixXf> kernels = {
        {ROBERTS, (Matrix2f() <<
          0, 1,
          -1, 0).finished()
        },
        {SOBEL, (Matrix3f() <<
          -1, 0, 1,
          -2, 0, 2,
          -1, 0, 1).finished()
        },
        {GAUSSIAN, (Matrix3f() <<
            1.0/16.0, 1.0/8.0, 1.0/16.0,
            1.0/8.0, 1.0/4.0, 1.0/8.0,
            1.0/16.0, 1.0/8.0, 1.0/16.0).finished()
        }
    };
};


#endif //IMAGEGRADIENTS_EDGE_DETECTOR_H