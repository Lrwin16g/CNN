#include "sigmoid.h"
#include "utils.h"

Sigmoid::Sigmoid(int batch_size, int input_size, int channel, int height, int width)
    : batch_size_(batch_size), input_size_(input_size), height_(height), width_(width),
      out_(NULL), out2d_(NULL)
{
    out_ = util::alloc<double>(batch_size_, input_size_);

    if (channel_ != 0 && height_ != 0 && width_ != 0) {
        out2d_ = util::alloc<double>(batch_size_, channel_, height_, width_);
    }
}

Sigmoid::~Sigmoid()
{
    util::free(out_, batch_size_);

    if (channel_ != 0 && height_ != 0 && width_ != 0) {
        util::free(out2d_, batch_size_, channel_, height_);
    }
}

void Sigmoid::forward(double const * const * const * const *input,
                      double ****output)
{
    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < channel_; ++j) {
            for (int k = 0; k < height_; ++k) {
                for (int l = 0; l < width_; ++l) {
                    out2d_[i][j][k][l] = util::sigmoid(input[i][j][k][l]);
                    output[i][j][k][l] = out2d_[i][j][k][l];
                }
            }
        }
    }
}

void Sigmoid::forward(double const * const *input, double **output)
{
    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            out_[i][j] = util::sigmoid(input[i][j]);
            output[i][j] = out_[i][j];
        }
    }
}

void Sigmoid::backward(double const * const * const * const *dout,
                       double ****dx)
{
    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < channel_; ++j) {
            for (int k = 0; k < height_; ++k) {
                for (int l = 0; l < width_; ++l) {
                    dx[i][j][k][l] = dout[i][j][k][l] * (1.0 - out2d_[i][j][k][l])
                                     * out2d_[i][j][k][l];
                }
            }
        }
    }
}

void Sigmoid::backward(double const * const *dout, double **dx)
{
    for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            dx[i][j] = dout[i][j] * (1.0 - out_[i][j]) * out_[i][j];
        }
    }
}