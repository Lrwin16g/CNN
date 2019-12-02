#include "relu.h"
#include "utils.h"

ReLU::ReLU(int batch_size, int input_size, int channel, int height, int width)
    : batch_size_(batch_size), input_size_(input_size), channel_(channel), 
      height_(height), width_(width), mask_(NULL), mask_2d_(NULL)
{
    mask_ = util::alloc<double>(batch_size_, input_size_);

    if (channel_ != 0 && height_ != 0 && width_ != 0) {
        mask_2d_ = util::alloc<double>(batch_size_, channel_, height_, width_);
    }
}

ReLU::~ReLU()
{
    util::free(mask_, batch_size_);

    if (channel_ != 0 && height_ != 0 && width_ != 0) {
        util::free(mask_2d_, batch_size_, channel_, height_);
    }
}

void ReLU::forward(double const * const * const * const *input,
                   double ****output)
{
    for (int n = 0; n < batch_size_; ++n) {
        for (int c = 0; c < channel_; ++c) {
            for (int y = 0; y < height_; ++y) {
                for (int x = 0; x < width_; ++x) {
                    if (input[n][c][y][x] <= 0.0) {
                        mask_2d_[n][c][y][x] = true;
                        output[n][c][y][x] = 0.0;
                    } else {
                        mask_2d_[n][c][y][x] = false;
                        output[n][c][y][x] = input[n][c][y][x];
                    }
                }
            }
        }
    }
}

void ReLU::forward(double const * const *input, double **output)
{
    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            if (input[i][j] <= 0.0) {
                mask_[i][j] = true;
                output[i][j] = 0.0;
            } else {
                mask_[i][j] = false;
                output[i][j] = input[i][j];
            }
        }
    }
}

void ReLU::backward(double const * const * const * const *dout,
                    double ****dx)
{
    for (int n = 0; n < batch_size_; ++n) {
        for (int c = 0; c < channel_; ++c) {
            for (int y = 0; y < height_; ++y) {
                for (int x = 0; x < width_; ++x) {
                    if (mask_2d_[n][c][y][x]) {
                        dx[n][c][y][x] = 0.0;
                    } else {
                        dx[n][c][y][x] = dout[n][c][y][x];
                    }
                }
            }
        }
    }
}

void ReLU::backward(double const * const *dout, double **dx)
{
    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            if (mask_[i][j]) {
                dx[i][j] = 0.0;
            } else {
                dx[i][j] = dout[i][j];
            }
        }
    }
}