
#pragma once

#include <opencv2/core.hpp>

typedef cv::Mat mmwrap_Matrix;
typedef cv::Size mmwrap_Size;

static void mmwrap_clear(mmwrap_Matrix& matrix) {
    matrix.setTo(cv::Scalar::all(0));
}

inline unsigned char mmwrap_saturate_cast_to_uchar(double x) {
    int v = int(std::round(x));
    return ((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0);
}

