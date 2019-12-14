#include "dropout.h"
#include "utils.h"

Dropout::Dropout(int batch_size, int input_size, int channel, int height, int width,
                 double ratio)
    : batch_size_(batch_size), input_size_(input_size), channel_(channel),
      height_(height), width_(width), ratio_(ratio)
{
    mask_ = util::alloc<bool>(batch_size_, input_size_);

    if (channel_ != 0 && height_ != 0 && width_ != 0) {
        mask2d_ = util::alloc<bool>(batch_size_, channel_, height_, width_);
    }
}

Dropout::~Dropout()
{
    util::free(mask_, batch_size_);

    if (channel_ != 0 && height_ != 0 && width_ != 0) {
        util::free(mask2d_, batch_size_, channel_, height_);
    }
}

void Dropout::forward(double const * const * const * const *input,
                      double ****output, bool train_flg)
{
    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < channel_; ++j) {
            for (int k = 0; k < height_; ++k) {
                for (int l = 0; l < width_; ++l) {
                    if (train_flg) {
                        mask2d_[i][j][k][l] = (util::randu<double>(0.0, 1.0) > ratio_);
                        output[i][j][k][l] = input[i][j][k][l] * static_cast<double>(mask2d_[i][j][k][l]);
                    } else {
                        output[i][j][k][l] = input[i][j][k][l] * (1.0 - ratio_);
                    }
                }
            }
        }
    }
}

void Dropout::forward(double const * const *input, double **output,
                      bool train_flg)
{
    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            if (train_flg) {
                mask_[i][j] = (util::randu<double>(0.0, 1.0) > ratio_);
                output[i][j] = input[i][j] * static_cast<double>(mask_[i][j]);
            } else {
                output[i][j] = input[i][j] * (1.0 - ratio_);
            }
        }
    }
}

void Dropout::backward(double const * const * const * const *dout,
                       double ****dx)
{
    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < channel_; ++j) {
            for (int k = 0; k < height_; ++k) {
                for (int l = 0; l < width_; ++l) {
                    dx[i][j][k][l] = dout[i][j][k][l] * static_cast<double>(mask2d_[i][j][k][l]);
                }
            }
        }
    }
}

void Dropout::backward(double const * const *dout, double **dx)
{
    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            dx[i][j] = dout[i][j] * static_cast<double>(mask_[i][j]);
        }
    }
}
