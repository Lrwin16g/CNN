#include "utils.h"
#include <cmath>

double** util::alloc(int dim_1, int dim_2)
{
    double **data = new double*[dim_1];
    for (int i = 0; i < dim_1; ++i) {
        data[i] = new double[dim_2];
        for (int j = 0; j < dim_2; ++j) {
            data[i][j] = 0.0;
        }
    }
    return data;
}

double**** util::alloc(int dim_1, int dim_2, int dim_3, int dim_4)
{
    double ****data = new double***[dim_1];
    for (int i = 0; i < dim_1; ++i) {
        data[i] = new double**[dim_2];
        for (int j = 0; j < dim_2; ++j) {
            data[i][j] = new double*[dim_3];
            for (int k = 0; k < dim_3; ++k) {
                data[i][j][k] = new double[dim_4];
                for (int l = 0; l < dim_4; ++l) {
                    data[i][j][k][l] = 0.0;
                }
            }
        }
    }
    return data;
}

void util::free(double **data, int dim_1)
{
    for (int i = 0; i < dim_1; ++i) {
        delete[] data[i];
    }
    delete[] data;
    data = NULL;
}

void util::free(double ****data, int dim_1, int dim_2, int dim_3)
{
    for (int i = 0; i < dim_1; ++i) {
        for (int j = 0; j < dim_2; ++j) {
            for (int k = 0; k < dim_3; ++k) {
                delete[] data[i][j][k];
            }
            delete[] data[i][j];
        }
        delete[] data[i];
    }
    delete[] data;
    data = NULL;
}

void util::dot(double const * const *lhs, double const * const *rhs, double **dst,
               int lhs_rows, int lhs_cols, int rhs_rows, int rhs_cols)
{
    for (int i = 0; i < lhs_rows; ++i) {
        for (int j = 0; j < rhs_cols; ++j) {
            dst[i][j] = 0.0;
            for (int k = 0; k < lhs_cols; ++k) {
                dst[i][j] += lhs[i][k] * rhs[k][j];
            }
        }
    }
}

void util::add(double const * const *lhs, const double *rhs, double **dst,
               int lhs_rows, int lhs_cols)
{
    for (int i = 0; i < lhs_rows; ++i) {
        for (int j = 0; j < lhs_cols; ++j) {
            dst[i][j] = lhs[i][j] * rhs[j];
        }
    }
}

void util::im2col(double const * const * const * const *src, double **dst,
                  int batch_size, int channel, int height, int width,
                  int filter_h, int filter_w, int stride = 1, int pad = 0)
{
    int output_h = (height + 2 * pad - filter_h) / stride + 1;
    int output_w = (width + 2 * pad - filter_w) / stride + 1;

    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < channel; ++c) {
            for (int y = -pad; y <= height + pad - filter_h; y += stride) {
                for (int x = -pad; x <= width + pad - filter_w; x += stride) {
                    int dst_y = n * (output_h * output_w) + ((y + pad) / stride) * output_w
                                + ((x + pad) / stride);
                    for (int v = 0; v < filter_h; ++v) {
                        for (int u = 0; u < filter_w; ++u) {
                            int src_y = y + v;
                            int src_x = x + u;
                            int dst_x = c * (filter_h * filter_w) + v * filter_w + u;

                            if (0 <= src_y && src_y < height && 0 <= src_x && src_x < width) {
                                dst[dst_y][dst_x] = src[n][c][src_y][src_x];
                            } else {
                                dst[dst_y][dst_x] = 0.0;
                            }
                        }
                    }
                }
            }
        }
    }
}

void util::col2im(double const * const *src, double ****dst,
                  int batch_size, int channel, int height, int width,
                  int filter_h, int filter_w, int stride = 1, int pad = 0)
{
    int input_h = (height + 2 * pad - filter_h) / stride + 1;
    int input_w = (width + 2 * pad - filter_w) / stride + 1;

    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < channel; ++c) {
            for (int y = -pad; y <= height + pad - filter_h; y += stride) {
                for (int x = -pad; x <= width + pad - filter_w; x += stride) {
                    int src_y = n * (input_h * input_w) + ((y + pad) / stride) * input_w
                                + ((x + pad) / stride);
                    for (int v = 0; v < filter_h; ++v) {
                        for (int u = 0; u < filter_w; ++u) {
                            int src_x = c * (filter_h * filter_w) + v * filter_w + u;
                            int dst_y = y + v;
                            int dst_x = x + u;

                            if (0 <= dst_y && dst_y < height && 0 <= dst_x && dst_x < width) {
                                dst[n][c][dst_y][dst_x] = src[src_y][src_x];
                            }
                        }
                    }
                }
            }
        }
    }
}